"""
Baseten Cold Start Predictor & Warmer MVP
Replace BASETEN_API_KEY with your actual API key or set the BASETEN_API_KEY environment variable.

Note: For best Pylance support, install pandas-stubs and statsmodels-stubs:
    pip install pandas-stubs statsmodels-stubs
"""

# Import necessary libraries for typing, environment variables, HTTP requests, data handling, time, logging, and forecasting
from typing import Any, Dict, Optional
import os
from dotenv import load_dotenv
import requests
import pandas as pd
import time
import logging
from datetime import datetime, timedelta, timezone
import statsmodels.api as sm  # type: ignore
import warnings

# Logging setup for info and error messages
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Suppress and redirect external library warnings to logger
warnings.filterwarnings("always")
def custom_warning(
    message: str,
    category: type,
    filename: str,
    lineno: int,
    file: Optional[Any] = None,
    line: Optional[str] = None
) -> None:
    msg = str(message)
    if len(msg) > 80:
        msg = msg[:77] + '...'
    logger.warning(f"[LIBRARY WARNING] {category.__name__}: {msg}")
warnings.showwarning = custom_warning

# Load environment variables from .env file
load_dotenv()

# Get Baseten API key from environment variable
BASETEN_API_KEY: str = os.environ.get("BASETEN_API_KEY", "")
if not BASETEN_API_KEY:
    logger.error("BASETEN_API_KEY environment variable not set. Exiting.")
    exit(1)

# Set your Baseten model ID here
TARGET_MODEL_ID: str = "zq8my5dw"  # <-- Replace with your model ID
# Construct the model's API URL
MODEL_API_URL: str = f"https://model-{TARGET_MODEL_ID}.api.baseten.co/environments/production/predict"
# Threshold for when to trigger a warm-up (requests/sec)
WARM_UP_THRESHOLD: float = 0.5  # requests/sec
# URL to fetch metrics from Baseten
METRICS_URL: str = "https://app.baseten.co/metrics"
# How many hours of data to keep in memory
ROLLING_WINDOW_HOURS: int = 48  # Keep last 48 hours of data
# How many future time steps to forecast
FORECAST_HORIZON: int = 12  # Number of future steps to forecast
# How often to resample data (in minutes)
RESAMPLE_MINUTES: int = 5  # Resample interval in minutes
# Minimum data points required for prediction
MIN_DATA_POINTS_FOR_PREDICTION: int = 5 
# Heuristic for robust SARIMAX performance (e.g., 5 hours of 5-min data)
SARIMAX_ROBUST_DATA_THRESHOLD: int = 60

# Global flag to ensure SARIMAX low-data warning is only logged once
_sarimax_low_data_warning_issued: bool = False


def fetch_baseten_metrics(api_key: str) -> Dict[str, Any]:
    """
    Fetches metrics from Baseten for the specified model.
    Returns a dictionary with timestamp, request_rate, and active_replicas.
    Handles rate limits and server errors gracefully.
    """
    headers: Dict[str, str] = {"Authorization": f"Api-Key {api_key}"}
    try:
        response: requests.Response = requests.get(METRICS_URL, headers=headers, timeout=10)
        if response.status_code == 429:
            logger.warning("[METRICS] Rate limit exceeded (429). Skipping this cycle.")
            return {}
        if response.status_code >= 500:
            logger.error(f"[METRICS ERROR] Server error: {response.status_code}. Skipping this cycle.")
            return {}
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"[METRICS ERROR] Exception during metrics fetch: {e}")
        return {}
    # Parse the metrics response line by line
    lines: list[str] = response.text.splitlines()
    request_rate: Optional[float] = None
    active_replicas: Optional[int] = None
    for line in lines:
        # Look for the request rate metric for the target model
        if f'baseten_inference_requests_total' in line and f'model_id="{TARGET_MODEL_ID}"' in line:
            try:
                value_str: str = line.split()[-1]
                request_rate = float(value_str)
            except Exception:
                logger.warning(f"[METRICS PARSE] Failed to parse request_rate from line: {line}")
                continue
        # Look for the active replicas metric for the target model
        if f'baseten_replicas_active' in line and f'model_id="{TARGET_MODEL_ID}"' in line:
            try:
                value_str = line.split()[-1]
                active_replicas = int(float(value_str))
            except Exception:
                logger.warning(f"[METRICS PARSE] Failed to parse active_replicas from line: {line}")
                continue
    # If either metric is missing, log a warning
    if request_rate is None or active_replicas is None:
        logger.warning(
            f"[METRICS MISSING] No metrics found for model_id={TARGET_MODEL_ID}. "
            "This may indicate no recent traffic or a metrics API issue."
        )
        return {}
    # logger.info(f"[METRICS] request_rate={request_rate}, active_replicas={active_replicas}")
    return {
        "timestamp": datetime.now(timezone.utc),
        "request_rate": request_rate,
        "active_replicas": active_replicas
    }


def update_historical_data(new_metrics: Dict[str, Any], historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends new metrics to the historical DataFrame and trims data to the rolling window.
    Returns the updated DataFrame.
    """
    if not new_metrics:
        # logger.info("[DATA] No new metrics to append to historical data.")
        return historical_df
    # Create a new row from the latest metrics
    new_row: pd.DataFrame = pd.DataFrame({
        "timestamp": [new_metrics["timestamp"]],
        "request_rate": [new_metrics["request_rate"]],
        "active_replicas": [new_metrics["active_replicas"]]
    })
    # Concatenate the new row to the historical data
    updated_df: pd.DataFrame = pd.concat([historical_df, new_row], ignore_index=True)
    # Remove data older than the rolling window
    cutoff: datetime = datetime.now(timezone.utc) - timedelta(hours=ROLLING_WINDOW_HOURS)
    updated_df = updated_df[updated_df["timestamp"] >= cutoff]
    updated_df = updated_df.reset_index(drop=True)
    # logger.info(f"[DATA] Historical data updated. Rows: {len(updated_df)}")
    return updated_df


def predict_cold_starts(historical_data: pd.DataFrame, forecast_horizon: int = FORECAST_HORIZON) -> pd.Series:
    """
    Uses SARIMAX time series forecasting to predict future request rates.
    Returns a forecasted pandas Series for the next forecast_horizon steps.
    """
    global _sarimax_low_data_warning_issued
    if historical_data.empty or len(historical_data) < MIN_DATA_POINTS_FOR_PREDICTION:
        logger.warning(
            f"[PREDICTION] Insufficient data: Only {len(historical_data)} data points available. "
            f"At least {MIN_DATA_POINTS_FOR_PREDICTION} are required for prediction."
        )
        return pd.Series(dtype="float")
    # One-time info log about SARIMAX warnings when running with limited data
    if (len(historical_data) >= MIN_DATA_POINTS_FOR_PREDICTION and
        len(historical_data) < SARIMAX_ROBUST_DATA_THRESHOLD and
        not _sarimax_low_data_warning_issued):
        logger.info(
            f"[PREDICTION] Running SARIMAX with limited historical data ({len(historical_data)} data points). "
            f"You may see 'Too few observations' warnings from the statsmodels library. "
            f"These warnings typically subside with more data (e.g., >{SARIMAX_ROBUST_DATA_THRESHOLD} data points "
            f"or a full seasonal cycle for more robust estimation)."
        )
        _sarimax_low_data_warning_issued = True
    df: pd.DataFrame = historical_data.copy()
    # Set timestamp as index for resampling
    df = df.set_index("timestamp")
    # Resample to regular intervals and forward-fill missing values
    df = df.resample(f"{RESAMPLE_MINUTES}min").mean().ffill()
    # Calculate seasonal period (e.g., daily seasonality)
    s: int = int(60 / RESAMPLE_MINUTES * 24)
    try:
        # Fit SARIMAX model to the request rate
        model: Any = sm.tsa.statespace.SARIMAX(
            df["request_rate"], order=(1,1,1), seasonal_order=(1,1,1,s),
            enforce_stationarity=False, enforce_invertibility=False
        )
        model_fit: Any = model.fit(disp=False)
        forecast: pd.Series = model_fit.forecast(steps=forecast_horizon)
        # logger.info(f"[PREDICTION] Forecasted next {forecast_horizon} steps: {forecast.values}")
        return forecast
    except Exception as e:
        logger.error(f"[PREDICTION ERROR] SARIMAX prediction failed: {e}")
        return pd.Series(dtype="float")


def trigger_warming(forecast: pd.Series, current_active_replicas: int, model_api_url: str, api_key: str, warm_up_threshold: float = WARM_UP_THRESHOLD) -> None:
    """
    Checks the forecast for potential cold starts and sends a warm-up request if needed.
    Only triggers if predicted request rate exceeds threshold and there is only one active replica.
    """
    if forecast.empty:
        # logger.info("[WARM-UP] No forecast available, skipping warming check.")
        return
    # Check the first 3 forecasted intervals for cold start risk
    for i, predicted_rate_raw in enumerate(forecast.iloc[:3]):
        predicted_rate: float = float(predicted_rate_raw)
        if predicted_rate > warm_up_threshold and current_active_replicas <= 1:
            # If a cold start is likely, send a warm-up request
            timestamp: str = (datetime.now(timezone.utc) + timedelta(minutes=RESAMPLE_MINUTES * (i+1))).isoformat()
            logger.warning(f"[WARM-UP] Predicted cold start for {TARGET_MODEL_ID} at {timestamp}. Sending warm-up request.")
            headers: Dict[str, str] = {"Authorization": f"Api-Key {api_key}", "Content-Type": "application/json"}
            payload: Dict[str, Any] = {
                "messages": [
                    {"role": "user", "content": "ping"}
                ],
                "max_tokens": 1
            }
            try:
                resp: requests.Response = requests.post(model_api_url, headers=headers, json=payload, timeout=10)
                if resp.status_code == 200:
                    logger.info(f"[WARM-UP] Warm-up request sent successfully for {TARGET_MODEL_ID} at {timestamp}.")
                else:
                    logger.error(f"[WARM-UP ERROR] Warm-up request failed for {TARGET_MODEL_ID} at {timestamp}: {resp.status_code} {resp.text}")
            except requests.RequestException as e:
                logger.error(f"[WARM-UP ERROR] Exception during warm-up request for {TARGET_MODEL_ID} at {timestamp}: {e}")
            break
    else:
        # logger.info("[WARM-UP] No cold start predicted in the next 3 intervals.")
        pass


def main() -> None:
    """
    Main loop: fetches metrics, updates historical data, predicts cold starts, and triggers warming as needed.
    Runs indefinitely, sleeping between cycles to respect API rate limits.
    """
    # Initialize empty DataFrame for historical metrics
    historical_data: pd.DataFrame = pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
        "request_rate": pd.Series(dtype="float"),
        "active_replicas": pd.Series(dtype="int")
    })
    logger.info("[INFO] Baseten Cold Start Predictor & Warmer started.")
    while True:
        try:
            # Fetch latest metrics
            metrics: Dict[str, Any] = fetch_baseten_metrics(BASETEN_API_KEY)
            # Update historical data with new metrics
            historical_data = update_historical_data(metrics, historical_data)
            # Predict future request rates
            forecast: pd.Series = predict_cold_starts(historical_data)
            # Get current number of active replicas
            current_active_replicas: int = int(metrics["active_replicas"]) if metrics else 0
            # Trigger warming if a cold start is predicted
            trigger_warming(forecast, current_active_replicas, MODEL_API_URL, BASETEN_API_KEY, WARM_UP_THRESHOLD)
        except Exception as e:
            logger.critical(f"[CRITICAL ERROR] Unexpected error in main loop: {e}")
        # logger.info(f"[INFO] Sleeping for {60} seconds to respect API rate limit...")
        time.sleep(60)


if __name__ == "__main__":
    main() 