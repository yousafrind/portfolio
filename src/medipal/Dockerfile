FROM python:3.11-slim
WORKDIR /app
# install system deps for EasyOCR
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# download models at build time (can be moved to runtime if you prefer)
RUN python -c "from llm_med import download_once; download_once()"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]