FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY main.py agent_main.py agent_tools.py model_runner.py preprocessing.py web_agent.py ./
COPY web/ ./web/

COPY models/tv2/inference_config.json models/tv2/normalization.json models/tv2/preprocess_config.json ./models/tv2/
COPY models/tv2/exports/cnn_rr_fp32.onnx ./models/tv2/exports/

COPY check_ds2_records.py ./
COPY reports/tv2/ds2_deployed_record_counts.csv ./reports/tv2/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
