FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3.9 -

# Set Poetry to not create virtual environment
ENV POETRY_VENV_IN_PROJECT=false
ENV POETRY_NO_INTERACTION=1

# Set working directory
WORKDIR /app

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# Install dependencies with GPU support
RUN poetry install --with gpu --no-root

# Copy source code
COPY . .

# Install the package
RUN poetry install --with gpu

# Create data directories
RUN mkdir -p data/{raw,interim,processed,predictions}

# Set default command
ENTRYPOINT ["poetry", "run", "wtcell"]
CMD ["--help"] 