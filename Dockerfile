FROM python:3.11-slim

WORKDIR /app

# Copy files
COPY requirements.txt .
COPY app_fastapi.py .
COPY logistic_regression_model.pkl .
COPY feature_names.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]