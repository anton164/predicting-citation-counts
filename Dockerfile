FROM python:3.7
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY app.py ./app.py
COPY main.py ./main.py
COPY experiment_selection.py ./experiment_selection.py
COPY data_tools ./data_tools
COPY experiments ./experiments
COPY correlation_study.py ./correlation_study.py
COPY distribution_study.py ./distribution_study.py
COPY vectorize_text_study.py ./vectorize_text_study.py
COPY sample_data.jsonl ./sample_data.jsonl
CMD streamlit run app.py --server.port 80
