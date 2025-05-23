# Use CUDA base image for GPU inference
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONWARNINGS="ignore"

# Install system dependencies
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get install -y python3.10
RUN apt-get install -y python3.10-distutils
RUN apt-get install -y python3.10-venv
RUN apt-get install -y python3-pip
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get -f install --fix-missing
RUN apt-get clean

# Copy files to the container
COPY ./ ./

# Create venv and install packages inside it
RUN python3.10 -m venv /app/.venv
RUN /app/.venv/bin/pip install --upgrade pip
RUN /app/.venv/bin/pip install -r requirements.txt

# Set working directory
WORKDIR /app

# Set entrypoint for inference
ENTRYPOINT ["/app/.venv/bin/python", "results.py"]
