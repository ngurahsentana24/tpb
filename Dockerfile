FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
# ARG FULL=1 untuk memasang PyTorch (LSTM penuh): docker build --build-arg FULL=1 -t tbp-lab .
ARG FULL=0
COPY requirements.txt requirements-full.txt ./
RUN if [ "$FULL" = "1" ]; then pip install -r requirements-full.txt; else pip install -r requirements.txt; fi
COPY . .
ENV PORT=8000
EXPOSE 8000
CMD gunicorn app:app --workers 1 --threads 4 --timeout 600 --bind 0.0.0.0:$PORT
