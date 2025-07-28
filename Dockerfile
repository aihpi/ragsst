FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY ragsst ./ragsst
COPY entrypoint.sh ./entrypoint.sh

# Create folders for persistence (these will be mounted as volumes)
RUN mkdir -p /app/data /app/vector_db /app/exports /app/log

EXPOSE 7860

CMD ["python", "app.py"] 