FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY data/ data/
COPY src/ src/
COPY api/ api/
RUN python -m src.train
EXPOSE 8004
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8004"]
