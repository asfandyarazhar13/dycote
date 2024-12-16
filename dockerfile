FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    gfortran \
    libblas-dev \
    liblapack-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run_pipeline.py .
COPY LD2011_2014.txt .  # If you want to run 'elec' dataset test, ensure data file is available

# By default run the sine dataset
CMD ["python", "run_pipeline.py", "--dataset_type", "sine"]
