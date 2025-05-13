# Use CUDA base image for GPU inference
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONWARNINGS="ignore"

# Install system dependencies
RUN apt-get update 
RUN apt-get install -y software-properties-common
RUN apt-get update 
RUN apt-get install -y python3.10 python3.10-distutils python3-pip
RUN apt-get -f install --fix-missing
RUN apt-get install -y libgl1-mesa-glx

# Copy files to the container
COPY ./ ./

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Set working directory
WORKDIR /app

# Set entrypoint for inference
ENTRYPOINT ["python3", "predict.py"]
