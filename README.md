# Baseten Cold Start Predictor & Warmer

Predict and prevent cold starts for Baseten models by monitoring request rates and automatically sending warm-up requests. Uses SARIMAX time series forecasting to anticipate traffic spikes and keep your model warm.

## 🚀 Quick Start

### Run with Pre-built Docker Image (Recommended)
1. **Pull the image:**
   ```bash
   docker pull <your-dockerhub-username>/baseten-warmer:latest
   ```
2. **Run interactively (prompts for API key/model ID if not set):**
   ```bash
   docker run -it <your-dockerhub-username>/baseten-warmer:latest
   ```
   Or, pass credentials:
   ```bash
   docker run -e BASETEN_API_KEY=your_api_key -e TARGET_MODEL_ID=your_model_id <your-dockerhub-username>/baseten-warmer:latest
   # or
   docker run --env-file .env <your-dockerhub-username>/baseten-warmer:latest
   ```

### Build and Run Docker Image Manually (Alternative)
1. **Clone the repo:**
   ```bash
   git clone https://github.com/minhkhoango/Baseten-ColdStart-Guardian.git
   cd Baseten-ColdStart-Guardian
   ```
2. **Build the image:**
   ```bash
   docker build -t baseten-warmer .
   ```
3. **Run:**
   ```bash
   docker run -it baseten-warmer
   # or with credentials
   docker run -e BASETEN_API_KEY=your_api_key -e TARGET_MODEL_ID=your_model_id baseten-warmer
   docker run --env-file .env baseten-warmer
   ```

## Configuration (Optional)
Set as environment variables or in a `.env` file:
```env
BASETEN_API_KEY=your_baseten_api_key_here
TARGET_MODEL_ID=your_model_id_here
```

## Features
- Real-time Baseten metrics
- SARIMAX-based traffic prediction
- Automatic warm-up requests
- Robust logging & error handling

## Development
- Format: `poetry run black .`
- Type check: `poetry run mypy cold_start_warmer.py`

## License
MIT 