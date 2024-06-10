import json

def save_data(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)

# Fungsi untuk memuat data dari file JSON
def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)