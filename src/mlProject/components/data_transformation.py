import os
from src.mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging
import numpy as np
from src.mlProject.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransformation:
    def __init__(self, config):
        self.config = config

    def train_test_spliting(self):
        # Load the data
        data = pd.read_csv(self.config.data_path)

        # Transform the target variable and create tenure groups
        data['Churn'] = np.where(data.Churn == 'Yes', 1, 0)
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
        data['tenure_group'] = pd.cut(data.tenure, range(1, 80, 12), right=False, labels=labels)
        
        # Store columns separately
        id_column = data.pop('customerID')
        tenure_column = data.pop('tenure')
        monthly_charge_column = data.pop('MonthlyCharges')
        total_charges_column = data.pop('TotalCharges')

        # Convert to numeric and handle errors
        data['MonthlyCharges'] = pd.to_numeric(monthly_charge_column, errors='coerce')
        data['TotalCharges'] = pd.to_numeric(total_charges_column, errors='coerce')

        # Identify categorical columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns

        # Create dummy variables for categorical columns
        data_dummies = pd.get_dummies(data, columns=categorical_columns)

        # Initialize and apply MinMaxScaler
        scaler = MinMaxScaler()

        # Handle non-numeric values by dropping rows with NaN in scaled columns
        data_dummies = data_dummies.dropna(subset=['MonthlyCharges', 'TotalCharges'])

        # Scale all numerical columns
        numeric_columns = ['MonthlyCharges', 'TotalCharges']
        data_dummies[numeric_columns] = scaler.fit_transform(data_dummies[numeric_columns])
        data_dummies = data_dummies.replace({True: 1, False: 0})
        # Split the data into training and test sets (0.75, 0.25 split)
        train, test = train_test_split(data_dummies, test_size=0.25, random_state=42)

        # Save the split datasets to CSV
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        # Log and print shapes of the resulting datasets
        logger.info("Split data into training and test sets")
        logger.info(f"Training set shape: {train.shape}")
        logger.info(f"Test set shape: {test.shape}")

        print(f"Training set shape: {train.shape}")
        print(f"Test set shape: {test.shape}")
