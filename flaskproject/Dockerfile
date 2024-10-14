# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /flaskproject

# Install FFmpeg and other dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg python3-pip && \
    apt-get install -y software-properties-common && \
    apt install -y ubuntu-drivers-common && \
    ubuntu-drivers devices && \
    apt-get install nvidia-driver-525 && \
    rm -rf /var/lib/apt/lists/*

# # Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .
EXPOSE 5000

# Specify the command to run your application
CMD ["flask", "run", "--host=0.0.0.0"]