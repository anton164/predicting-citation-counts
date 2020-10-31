# aml-research-project


## Set-up (Python 3.7+)
1. Install dependencies
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
2. Copy datasets to the repository manually
   - 250k.docs.jsonl (sample of 250k docs)
   - mag5.docs.jsonl (full dataset with 5 mill docs)

## Cloud instances 
- main branch (GCP)
- anton/main branch (GCP)
- jan/main branch (GCP)
- eli/main branch (GCP)

[IPs can be found here](https://console.cloud.google.com/compute/instances?project=stellar-mercury-292013) or by checking the final step of the github workflow.

## Run locally
```
streamlit run explore.py
```
