import pandas as pd

class UnivariateAnalyser:
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)
        self.quantitative = []
        self.qualitative = []

    def categorize_columns(self):
        # Using select_dtypes to select columns by data types
        self.quantitative = self.dataset.select_dtypes(include=['number']).columns.tolist()
        self.qualitative = self.dataset.select_dtypes(exclude=['number']).columns.tolist()
        
        return self.quantitative, self.qualitative
    
    
    def calculate_statistics(self):
        # Ensure quantitative columns are categorized
        if not self.quantitative:
            self.categorize_columns()

        # Create lists to hold statistics
        descriptive = pd.DataFrame(index=["Mean","Median","Mode"],columns=self.qualitative)

        # Calculate statistics for each quantitative column
        for col in self.quantitative:
            descriptive[col]["Mean"] = self.dataset[col].mean()
            descriptive[col]["Median"] = self.dataset[col].median()
            descriptive[col]["Mode"] = self.dataset[col].mode()[0]  # mode() returns a Series, get the first mode value
        
        return descriptive
