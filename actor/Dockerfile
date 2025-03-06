# First Stage: Build dependencies
FROM python:3.9-slim as builder

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for building
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Install the required Python packages (from requirements)
COPY ./actor/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Second Stage: Final image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files from the builder stage (excluding the build dependencies)
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Install only runtime system dependencies (no build dependencies)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy the application code into the container (from the host machine)
COPY ./actor /app

# Run the trainer script
CMD ["python", "actor.py"]
