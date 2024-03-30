import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_excel(file_path)
        return data
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def analyze_data(data):
    """Perform basic data analysis."""
    if data is not None:
        # Display summary statistics
        print("Summary Statistics:")
        print(data.describe())

        # Plot histograms for numeric columns
        print("Histograms:")
        for col in data.select_dtypes(include=['int', 'float']):
            data[col].plot(kind='hist', bins=10)
            plt.title(col)
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.show()
        
        # Plot bar plot for the class label (string type)
        class_label_counts = data['Class'].value_counts()
        class_label_counts.plot(kind='bar')
        plt.title('Class Label Distribution')
        plt.xlabel('Class Label')
        plt.ylabel('Count')
        plt.show()

def main():
    file_path = "DryBeanDataset\Dry_Bean_Dataset.xlsx" # DryBeanDataset\Dry_Bean_Dataset.xlsx
    data = load_data(file_path)
    analyze_data(data)

    # Get roll number
    roll_number = int(input("Enter your roll number: "))

    if roll_number % 2 == 0:
        # List out missing values
        missing_values = data.isnull().sum()
        print("Missing values:")
        print(missing_values)
    else:
        # Encode categorical values
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            data[col] = data[col].astype('category').cat.codes
        print("Categorical values encoded.")

if __name__ == "__main__":
    main()
