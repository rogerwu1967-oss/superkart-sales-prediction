FROM python:3.13-slim

# libgomp1 is the OpenMP runtime required by XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory to the container's working directory
COPY . .

# Install dependencies from the requirements file without using cache to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Start the application using Gunicorn with 4 worker processes on port 7860
# - `app:superkart_api`: app.py contains the Flask instance named `superkart_api`
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:superkart_api"]
