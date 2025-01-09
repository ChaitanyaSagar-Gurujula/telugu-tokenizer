FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ .
COPY telugu_tokenizer_vocab.json .
COPY telugu_tokenizer_merges.json .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"] 