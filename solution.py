import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load datasets
path = "datasets/"
training_tokens = pd.read_parquet(path + "training_tokens.parquet")
transactions = pd.read_parquet(path + "transactions.parquet")
token_transfers = pd.read_parquet(path + "token_transfers.parquet")
nft_transfers = pd.read_parquet(path + "nft_transfers.parquet")
dex_swaps = pd.read_parquet(path + "dex_swaps.parquet")
test_tokens = pd.read_parquet(path + "test_tokens.parquet")

# 1. Feature Engineering
def preprocess_data(training_tokens, transactions, token_transfers, nft_transfers, dex_swaps):
    # Example features
    # Transaction-based features
    tx_counts = transactions.groupby('TO_ADDRESS').size().reset_index(name='tx_count')
    tx_value_sum = transactions.groupby('TO_ADDRESS')['VALUE'].sum().reset_index(name='tx_value_sum')

    # Transfer-based features
    transfer_counts = token_transfers.groupby('Contract_Address').size().reset_index(name='transfer_count')
    transfer_value_sum = token_transfers.groupby('Contract_Address')['Amount_Precise'].sum().reset_index(name='transfer_value_sum')

    print(training_tokens)
    # Combine features
    features = training_tokens.merge(tx_counts, left_on='ADDRESS', right_on='TO_ADDRESS', how='left')
    features = features.merge(tx_value_sum, left_on='ADDRESS', right_on='TO_ADDRESS', how='left')
    features = features.merge(transfer_counts, left_on='ADDRESS', right_on='Contract_Address', how='left')
    features = features.merge(transfer_value_sum, left_on='ADDRESS', right_on='Contract_Address', how='left')
    
    # Replace missing values with 0
    features.fillna(0, inplace=True)

    return features

# Generate features
features = preprocess_data(training_tokens, transactions, token_transfers, nft_transfers, dex_swaps)

# Separate labels
X = features.drop(columns=['LABEL', 'ADDRESS'])
y = features['LABEL']

# 2. Data Splitting
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# 4. Evaluation
y_val_pred = rf_model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

# 5. Test Set Predictions
test_features = preprocess_data(test_tokens, transactions, token_transfers, nft_transfers, dex_swaps)
X_test = test_features.drop(columns=['ADDRESS'])
test_predictions = rf_model.predict(X_test)

# 6. Create Submission File
submission = pd.DataFrame({
    'ADDRESS': test_features['ADDRESS'],
    'PRED': test_predictions
})
submission.to_csv("submission.csv", index=False)
