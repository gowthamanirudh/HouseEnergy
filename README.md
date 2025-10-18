# Household Energy Consumption Prediction API

A dockerized machine learning API for predicting household energy consumption using an optimized Random Forest model.

## 📊 Project Overview

This project provides a REST API for predicting daily household energy consumption based on historical patterns. The model was trained on 2+ million records of household power consumption data and achieves:

- **RMSE**: 0.231
- **MAE**: 0.171
- **R²**: 0.489

## 🚀 Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and run the container
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Using Docker

```bash
# Build the image
docker build -t household-energy-api .

# Run the container
docker run -d -p 8000:8000 --name energy-api household-energy-api

# Check logs
docker logs -f energy-api

# Stop the container
docker stop energy-api
docker rm energy-api
```

### Local Development (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
python app.py
```

## 📡 API Endpoints

Once running, the API will be available at `http://localhost:8000`

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Available Endpoints

#### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### 2. Make Prediction
```bash
POST /predict
```

**Request Body:**
```json
{
  "features": [1.5, 1.6, 1.4, 1.7, 1.5, 1.3, 1.8]
}
```

**Response:**
```json
{
  "prediction": 1.52,
  "model_info": "Random Forest Regressor (Optimized & Pruned)"
}
```

#### 3. Model Information
```bash
GET /model/info
```

**Response:**
```json
{
  "model_type": "RandomForestRegressor",
  "n_estimators": 200,
  "max_depth": 20,
  "min_samples_split": 5,
  "n_features": 7
}
```

## 🧪 Testing the API

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [1.5, 1.6, 1.4, 1.7, 1.5, 1.3, 1.8]}'

# Get model info
curl http://localhost:8000/model/info
```

### Using Python

```python
import requests

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [1.5, 1.6, 1.4, 1.7, 1.5, 1.3, 1.8]}
)
print(response.json())
```

## 📁 Project Structure

```
HouseEnergy/
├── app.py                              # FastAPI application
├── random_forest_best_pruned.pkl       # Trained model
├── Household_Energy.ipynb              # Model training notebook
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Docker configuration
├── docker-compose.yml                  # Docker Compose configuration
├── .dockerignore                       # Docker ignore file
└── README.md                           # This file
```

## 🛠️ Technology Stack

- **Framework**: FastAPI
- **ML Library**: scikit-learn
- **Model**: Random Forest Regressor (optimized & pruned)
- **Server**: Uvicorn
- **Containerization**: Docker

## 📝 Model Details

The model uses lag features (past energy consumption values) to predict future consumption:
- **Input**: 7 lag features (previous days' consumption)
- **Output**: Predicted energy consumption for the target day
- **Optimization**: Hyperparameter tuning via RandomizedSearchCV
- **Feature Selection**: Removed features with <1% importance

## 🔧 Configuration

### Environment Variables

You can customize the API by setting environment variables:

```bash
# In docker-compose.yml or when running docker
environment:
  - PORT=8000
  - WORKERS=1
```

### Port Configuration

To change the port, modify:
- `docker-compose.yml`: Update the ports mapping
- `Dockerfile`: Update the EXPOSE and CMD instructions

## 🐛 Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs

# Rebuild without cache
docker-compose build --no-cache
docker-compose up
```

### Model not loading
Ensure `random_forest_best_pruned.pkl` is in the same directory as `app.py`

### Port already in use
```bash
# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use port 8001 instead
```

## 📄 License

This project is for educational and demonstration purposes.

## 👤 Author

Anirudh

## 🤝 Contributing

Feel free to submit issues and enhancement requests!
