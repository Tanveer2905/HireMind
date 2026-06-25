FROM python:3.11-slim

# Install system dependencies needed to compile AI libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face requires running as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy all project files into the container
COPY --chown=user . $HOME/app

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Download all models (LLM, Embeddings, SpaCy) during the build phase
# This ensures the space starts instantly without downloading 5GB every time
RUN python download_models.py

# Hugging Face Spaces route traffic to port 7860
EXPOSE 7860

# Start the Flask app on port 7860
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]
