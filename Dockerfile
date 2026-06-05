FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

# Download NLTK resources
RUN python download_nltk.py

EXPOSE 8501

CMD ["streamlit", "run", "app/frontend.py", "--server.address=0.0.0.0"]