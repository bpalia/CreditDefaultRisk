# Start from a base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install libgomp1 to run LGBM models
RUN apt-get update && apt-get install -y libgomp1

# Install the required packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application code into the container
COPY ["tuned_100model_full.joblib", "home_credit_api.py", "./"] .

# Expose the app port
EXPOSE 80

# Run command
CMD ["uvicorn", "home_credit_api:app", "--host", "0.0.0.0", "--port", "80"]