# aml-research-project

GCP deployment address for the main branch:
http://35.209.145.171


## Set-up (Python 3.7+)
1. Install dependencies
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
2. Copy datasets to the repository manually
   - 250k.docs.jsonl (sample of 250k docs)
   - mag5.docs.jsonl (full dataset with 5 mill docs)

## Run locally

```
streamlit run app.py
```

*Or run individual streamlit pages:*

- **Initial studies:** `streamlit run explore.py`
- **Dataframe (feature) selection:** `streamlit run main.py`
- **Experiment selection:** `streamlin run experiment_selection.py`


