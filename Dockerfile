FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
ENV PORT=8501
EXPOSE 8501
CMD ["streamlit", "run", "fenxi/omni_engine_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]