import pandas as pd
import numpy as np

class UnivariateAnalyser:
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)
        self.original_dataset = self.dataset.copy()  # Save a copy of the original dataset
        self.quantitative = []
        self.qualitative = []
        self.outliers = {
            'lesser_outliers': [],
            'greater_outliers': []
        }
        self.replaced_dataset = self.dataset.copy()

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
        descriptive = pd.DataFrame(index=["Mean", "Median", "Mode", "Q1:25%", "Q2:50%", "Q3:75%", "Q4:100%","IQR","1.5Rule","LesserOutlier","GreaterOutlier","Min","Max","Skew","Kurtosis"], columns=self.quantitative)

        # Calculate statistics for each quantitative column
        for col in self.quantitative:
            descriptive.at["Mean", col] = self.dataset[col].mean()
            descriptive.at["Median", col] = self.dataset[col].median()
            descriptive.at["Mode", col] = self.dataset[col].mode().iloc[0] if not self.dataset[col].mode().empty else np.nan
            descriptive.at["Q1:25%", col] = np.percentile(self.dataset[col].dropna(), 25)
            descriptive.at["Q2:50%", col] = np.percentile(self.dataset[col].dropna(), 50)
            descriptive.at["Q3:75%", col] = np.percentile(self.dataset[col].dropna(), 75)
            descriptive.at["Q4:100%", col] = np.percentile(self.dataset[col].dropna(), 100)
            descriptive.at["IQR", col] = descriptive.at["Q3:75%", col] - descriptive.at["Q1:25%", col]
            descriptive.at["1.5Rule", col] = 1.5*descriptive.at["IQR", col]
            descriptive.at["LesserOutlier", col] = descriptive.at["Q1:25%", col]-descriptive.at["1.5Rule", col]
            descriptive.at["GreaterOutlier", col] = descriptive.at["Q3:75%", col]+descriptive.at["1.5Rule", col]
            descriptive.at["Min", col] = self.dataset[col].min()
            descriptive.at["Max", col] = self.dataset[col].max()
            descriptive.at["Skew", col] = self.dataset[col].skew()
            descriptive.at["Kurtosis", col] = self.dataset[col].kurtosis()
            
            
            # Store outliers by column
            # Check if there are outliers and store column names
            if any(self.dataset[col] < descriptive.at["LesserOutlier", col]):
                self.outliers['lesser_outliers'].append(col)
            if any(self.dataset[col] > descriptive.at["GreaterOutlier", col]):
                self.outliers['greater_outliers'].append(col)
        
        return descriptive
    
    def get_outliers(self):
        return self.outliers
    
    def replace_outliers(self):
        # Ensure quantitative columns are categorized
        if not self.quantitative:
            self.categorize_columns()

        # Replace outliers in the dataset
        for col in self.quantitative:
            lesser_threshold = self.calculate_statistics().at["LesserOutlier", col]
            greater_threshold = self.calculate_statistics().at["GreaterOutlier", col]
            
            # Replace lesser outliers with the lesser threshold
            self.replaced_dataset[col] = np.where(self.dataset[col] < lesser_threshold, lesser_threshold, self.dataset[col])

            # Replace greater outliers with the greater threshold
            self.replaced_dataset[col] = np.where(self.replaced_dataset[col] > greater_threshold, greater_threshold, self.replaced_dataset[col])
            
 
    
    def compare_datasets(self):
        # Initialize a list to hold the comparison results
        comparison_list = []

        # Iterate through each column in the quantitative dataset
        for col in self.quantitative:
            # Calculate statistics for the current column
            stats = self.calculate_statistics()

            # Find lesser and greater outliers in the original dataset
            lesser_outliers_original = self.dataset[self.dataset[col] < stats.at["LesserOutlier", col]]
            greater_outliers_original = self.dataset[self.dataset[col] > stats.at["GreaterOutlier", col]]

            # Find lesser and greater outliers in the replaced dataset
            lesser_outliers_replaced = self.replaced_dataset[self.replaced_dataset[col] < stats.at["LesserOutlier", col]]
            greater_outliers_replaced = self.replaced_dataset[self.replaced_dataset[col] > stats.at["GreaterOutlier", col]]

            # Append the comparison results for the current column to the list
            comparison_list.append({
                'Column': col,
                'Original Lesser Outliers Count': len(lesser_outliers_original),
                'Original Greater Outliers Count': len(greater_outliers_original),
                'Replaced Lesser Outliers Count': len(lesser_outliers_replaced),
                'Replaced Greater Outliers Count': len(greater_outliers_replaced),
                'Original Lesser Outliers': lesser_outliers_original[col].tolist(),
                'Original Greater Outliers': greater_outliers_original[col].tolist(),
                'Replaced Lesser Outliers': lesser_outliers_replaced[col].tolist(),
                'Replaced Greater Outliers': greater_outliers_replaced[col].tolist(),
                'Original Min': self.dataset[col].min(),
                'Original Max': self.dataset[col].max(),
                'Replaced Min': self.replaced_dataset[col].min(),
                'Replaced Max': self.replaced_dataset[col].max()
            })

        # Convert the list to a DataFrame
        comparison_df = pd.DataFrame(comparison_list)

        return comparison_df

    def get_replaced_dataset(self):
        return self.replaced_dataset

    def get_outliers(self):
        return self.outliers
    
    def frequency_analysis(self, column):
        """Calculate frequency, relative frequency, and cumulative frequency for a given column."""
        # Calculate frequency
        frequency = self.dataset[column].value_counts()
        
        # Calculate relative frequency
        relative_frequency = frequency / len(self.dataset)

        # Calculate cumulative frequency
        cumulative_frequency = relative_frequency.cumsum()

        # Combine into a DataFrame
        frequency_table = pd.DataFrame({
            'Frequency': frequency,
            'Relative Frequency': relative_frequency,
            'Cumulative Frequency': cumulative_frequency
        })

        return frequency_table

    def frequency_analysis_all_quantitative(self):
        """Calculate frequency, relative frequency, and cumulative frequency for all quantitative columns."""
        if not self.quantitative:
            self.categorize_columns()

        all_frequency_tables = {}

        for col in self.quantitative:
            all_frequency_tables[col] = self.frequency_analysis(col)

        return all_frequency_tables