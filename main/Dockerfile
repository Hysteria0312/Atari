FROM python:3.9-slim AS build

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libopenblas-dev \
    libomp-dev

FROM python:3.9-slim

WORKDIR /app

COPY --from=build /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

COPY . /app

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
