import subprocess
import os

files = [{
    "output_data_name": "distilbert.csv",
    "model_name": "distilbert-base-uncased",
    "target_text": "Title"
},
    {
    "output_data_name": "biobert.csv",
    "model_name": "michiyasunaga/BioLinkBERT-base",
    "target_text": "Title"
},
    {
    "output_data_name": "gptbert.csv",
    "model_name": "baykenney/bert-base-gpt2detector-topk40",
    "target_text": "Title"
}]
    
for file in files:
    cmd = f"python train.py --output_data_name {file["output_data_name"]} --model_name {file["model_name"]} --target_text {file["target_text"]}"
    subprocess.Popen(cmd, shell=True)

