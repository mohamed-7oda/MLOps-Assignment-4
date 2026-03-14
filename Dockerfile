# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run training script
CMD ["python", "train.py"]

# To build the image: docker build -t gan_project .
# To run the container: docker run --rm gan_project