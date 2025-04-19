import pandas as pd

class DataIO:
    @staticmethod
    def load_data(file_path):
        return pd.read_csv(file_path)
    
    @staticmethod
    def save_results(results, file_path):
        with open(file_path, 'w') as f:
            f.write(str(results))