# Baseten Cold Start Predictor & Warmer

This project predicts and prevents cold starts for Baseten models by monitoring request rates and automatically sending warm-up requests when needed. It uses time series forecasting (SARIMAX) to anticipate traffic spikes and proactively keeps your model warm.

## Features
- Fetches real-time metrics from Baseten
- Predicts future request rates using SARIMAX
- Automatically triggers warm-up requests to prevent cold starts
- Robust logging and error handling

## Requirements
- Python 3.12+
- [Poetry](https://python-poetry.org/) for dependency management

## Installation
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Baseten
   ```
2. **Install dependencies:**
   ```bash
   poetry install
   ```

## Configuration
1. **Environment Variables:**
   - Copy `.env.example` to `.env` and fill in your Baseten API key:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env`:
     ```env
     BASETEN_API_KEY=your_baseten_api_key_here
     ```
2. **Update Model ID:**
   - Edit `cold_start_warmer.py` and set `TARGET_MODEL_ID` to your Baseten model's ID.

## Usage
Run the predictor and warmer:
```bash
poetry run python cold_start_warmer.py
```

The script will continuously monitor your Baseten model and send warm-up requests as needed.

## Docker
A `Dockerfile` is provided for containerized deployment.

### Build the Docker image:
```bash
docker build -t baseten-warmer .
```

### Run the container:
```bash
docker run --env-file .env baseten-warmer
```

## Development
- Code formatting: `poetry run black .`
- Type checking: `poetry run mypy cold_start_warmer.py`

## License
MIT 