FROM python:3.9-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN export PYTHONUNBUFFERED=1

WORKDIR /app
COPY app /app


EXPOSE 7860

CMD ["python", "-u", "app.py"]
