# Use an official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
# Using the specific index for CUDA support as requested, though in Docker without GPU passthrough
# the model will run on CPU unless nvidia-runtime is configured.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
# This includes utils/, models.py, best_model.pth, and deploy/
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run the Streamlit app
# We point to deploy/app.py
CMD ["streamlit", "run", "deploy/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
