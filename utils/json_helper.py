import json

def json_open(data_path):
    with open(data_path) as f:
        data = json.load(f)
    
    return data