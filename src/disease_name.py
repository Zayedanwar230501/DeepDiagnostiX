import json

class DiseaseInfo:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.diseases = json.load(f)  # Fixed attribute name

    def get_info(self, disease_name):
        for disease in self.diseases:
            if disease['name'].lower() == disease_name.lower():
                return disease
        return None
