FROM python:3.11.7

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install libmagic-dev -y

RUN pip install -r requirements.txt

COPY . . 

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

EXPOSE 8501