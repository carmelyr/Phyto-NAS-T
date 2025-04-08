# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git

# Copy your project into the container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -e .  # Install your package with setup.py
RUN pip install scikit-learn numpy pandas

# Set default command (can be overridden in YAML)
CMD ["python", "run_kubernetes.py"]

