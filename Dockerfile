# Create lightweight python image
FROM python:3.12.7-slim

RUN apt-get update && \
    apt-get install -y \
    # Packages required for libraries which use C (Numpy etc.)
    build-essential \
    # Packages required for librosa
    libsndfile1 \
    libasound2-dev && \
    # Clean up
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Specify the working directory
WORKDIR /code

# Install dependencies first before copying code
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy everything into the working directory
# .dockerignore is used to filter out the audiofiles
COPY . .
EXPOSE 8000

# Start the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]