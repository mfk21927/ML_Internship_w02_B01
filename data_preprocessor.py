import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, path):
        self.path = path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.path)
        return self.data

    def handle_missing_values(self):
        # Numeric columns filled with median
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].median())
        
        # Categorical columns filled with mode
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])

    def encode_categorical(self, columns):
        le = LabelEncoder()
        for col in columns:
            self.data[col] = le.fit_transform(self.data[col])

    def scale_features(self, columns):
        scaler = StandardScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])

    def split_data(self, target_column, test_size=0.2):
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=42)

    def save_processed_data(self, output_path):
        self.data.to_csv(output_path, index=False)

if __name__ == "__main__":
    
    preprocessor = DataPreprocessor('titanic_cleaned.csv')
    
    # Loading data
    preprocessor.load_data()
    
    #  Handle missing values
    preprocessor.handle_missing_values()
    
    #  Encode Categorical (e.g., Sex and Embarked)
    preprocessor.encode_categorical(['Sex', 'Embarked', 'Title'])
    
    #  Scale Numerical (e.g., Age and Fare)
    preprocessor.scale_features(['Age', 'Fare'])
    
    #  Split Data
    X_train, X_test, y_train, y_test = preprocessor.split_data('Survived')
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    #  Save final processed data
    preprocessor.save_processed_data('titanic_final_processed.csv')
    print("Processed data saved successfully.")