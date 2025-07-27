# Use a slim Python base image for smaller size and AMD64 compatibility
# Explicitly specify platform as per hackathon docs [cite: 56, 57]
FROM --platform=linux/amd64 python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python script into the container
COPY main.py .

# Command to run your script when the container starts
CMD ["python", "main.py"]