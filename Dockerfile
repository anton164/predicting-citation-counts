FROM python:3.7
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY explore.py ./explore.py
COPY data_tools ./data_tools
COPY sample_data.jsonl ./sample_data.jsonl
CMD streamlit run explore.py --server.port 80
