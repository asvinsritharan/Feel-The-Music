# Use a lightweight base image
FROM python:3.13

# Set working directory inside container
WORKDIR /app

# Copy dependencies first (better layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (data + source)
COPY . .

# Pre-download your SentenceTransformer model so it's cached in the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-roberta-large-v1')"

# Expose Flask port
EXPOSE 9332

# Default command
CMD ["python", "src/app.py"]
