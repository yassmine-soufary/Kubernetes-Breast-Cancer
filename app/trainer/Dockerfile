FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

# Un Job exécute une tâche puis s'arrête : pas de port, pas de serveur
CMD ["python", "train.py"]
