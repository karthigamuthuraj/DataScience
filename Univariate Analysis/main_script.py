import pandas as pd
import numpy as np

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

        # Create a DataFrame to hold statistics
        descriptive = pd.DataFrame(index=["Mean", "Median", "Mode", "Q1:25%", "Q2:50%", "Q3:75%", "Q4:100%"], columns=self.quantitative)

        # Calculate statistics for each quantitative column
        for col in self.quantitative:
            descriptive.at["Mean", col] = self.dataset[col].mean()
            descriptive.at["Median", col] = self.dataset[col].median()
            descriptive.at["Mode", col] = self.dataset[col].mode().iloc[0] if not self.dataset[col].mode().empty else np.nan
            descriptive.at["Q1:25%", col] = np.percentile(self.dataset[col].dropna(), 25)
            descriptive.at["Q2:50%", col] = np.percentile(self.dataset[col].dropna(), 50)
            descriptive.at["Q3:75%", col] = np.percentile(self.dataset[col].dropna(), 75)
            descriptive.at["Q4:100%", col] = np.percentile(self.dataset[col].dropna(), 100)
        
        return descriptive

# Usage
file_path = "Placement.csv"
analyzer = UnivariateAnalyser(file_path)
quantitative, qualitative = analyzer.categorize_columns()

print("Quantitative Columns:", quantitative)
print("Qualitative Columns:", qualitative)

stats_df = analyzer.calculate_statistics()

print(stats_df)
