FROM python:3.7
EXPOSE $PORT
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY explore.py ./explore.py
COPY sample_data.jsonl ./sample_data.jsonl
COPY 250k.docs.jsonl ./250k.docs.jsonl
CMD streamlit run explore.py --server.port $PORT --server.enableCORS false