import pandas as pd

def load_csv(file_path, encoding='utf-8'):
    """Load a CSV file with the specified encoding."""
    try:
        data = pd.read_csv(file_path, encoding=encoding)
        return data
    except UnicodeDecodeError:
        raise ValueError(f"Failed to read the file with encoding: {encoding}")

def clean_data(data):
    """Remove rows with missing values and duplicate rows."""
    data = data.dropna()
    data = data.drop_duplicates()
    return data

def main():
    # File path to the CSV file
    file_path = '../data/oil_production_statistics.csv'
    
    # Attempt to load the CSV file with different encodings
    encodings = ['utf-8', 'latin1', 'cp1252']
    for encoding in encodings:
        try:
            data = load_csv(file_path, encoding=encoding)
            print(f"Successfully read the file with encoding: {encoding}")
            break
        except ValueError as e:
            print(e)
    else:
        print("All attempted encodings failed. Please check the file or try a different encoding.")
        return
    
    # Clean the data
    data = clean_data(data)
    
    # Display the first few rows of the cleaned data
    print(data.head())

if __name__ == "__main__":
    main()