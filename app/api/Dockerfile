FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# La version de l'application est gravée dans l'image :
#   docker build --build-arg APP_VERSION=2.0 -t bc-api:2.0 .
ARG APP_VERSION=1.0
ENV APP_VERSION=${APP_VERSION}

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
