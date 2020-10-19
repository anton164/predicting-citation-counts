FROM python:3.7
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
RUN python -m spacy download en
COPY utils.py ./utils.py
COPY feature_utils.py ./feature_utils.py
COPY text_utils.py ./text_utils.py
COPY explore.py ./explore.py
COPY correlation_study.py ./correlation_study.py
COPY data_tools ./data_tools
COPY distribution_study.py ./distribution_study.py
COPY vectorize_text_study.py ./vectorize_text_study.py
COPY sample_data.jsonl ./sample_data.jsonl
CMD streamlit run explore.py --server.port 80
