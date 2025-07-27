# Use a slim Python base image for smaller size and AMD64 compatibility
# Explicitly specify platform as per hackathon docs
FROM --platform=linux/amd64 python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for PyMuPDF (if needed, though usually not for wheels)
# and general build tools. Some PyMuPDF builds might link against system libraries.
# For 'slim-buster', often this is minimal.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgomp1 \
    build-essential \
    pkg-config \
    libfreetype6-dev \
    libpng-dev \
    # Clean up apt cache to keep image size small
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies, including CPU-only PyTorch from its wheel index
# We use --extra-index-url for PyTorch CPU wheels
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# --- Pre-download the Sentence Transformer model during build ---
# This is crucial for offline execution.
# Choose a small model like 'all-MiniLM-L6-v2' (approx 90MB)
ENV MODEL_NAME="all-MiniLM-L6-v2"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['MODEL_NAME'])"

# Copy your main processing script
COPY main.py .

# Command to run your script when the container starts
CMD ["python", "main.py"]