import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load datasets
path = "datasets/"
training_tokens = pd.read_parquet(path + "training_tokens.parquet")
transactions = pd.read_parquet(path + "transactions.parquet")
token_transfers = pd.read_parquet(path + "token_transfers.parquet")
nft_transfers = pd.read_parquet(path + "nft_transfers.parquet")
test_tokens = pd.read_parquet(path + "test_tokens.parquet")
dex_swaps = pd.read_parquet(path + "dex_swaps.parquet")  # **Added this line**

print("Columns in training_tokens:", training_tokens.columns)

def preprocess_data(tokens, transactions, token_transfers, nft_transfers, dex_swaps, is_training=True):
    # Transaction-based features
    tx_counts = transactions.groupby('TO_ADDRESS').size().reset_index(name='tx_count')
    tx_value_sum = transactions.groupby('TO_ADDRESS')['VALUE'].sum().reset_index(name='tx_value_sum')

    transfer_counts = token_transfers.groupby('CONTRACT_ADDRESS').size().reset_index(name='transfer_count')
    transfer_value_sum = token_transfers.groupby('CONTRACT_ADDRESS')['AMOUNT_PRECISE'].sum().reset_index(name='transfer_value_sum')

    features = tokens.merge(tx_counts, left_on='ADDRESS', right_on='TO_ADDRESS', how='left')
    features = features.merge(tx_value_sum, left_on='ADDRESS', right_on='TO_ADDRESS', how='left')
    features = features.merge(transfer_counts, left_on='ADDRESS', right_on='CONTRACT_ADDRESS', how='left')
    features = features.merge(transfer_value_sum, left_on='ADDRESS', right_on='CONTRACT_ADDRESS', how='left')

    numeric_columns = features.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    features[numeric_columns] = imputer.fit_transform(features[numeric_columns])

    timestamp_columns = features.select_dtypes(include=['datetime', 'datetime64[ns, UTC]', 'datetime64[ms, UTC]']).columns
    for col in timestamp_columns:
        features[col] = pd.to_datetime(features[col], errors='coerce')
        features[col] = features[col].astype(np.int64) // 10**9

    features = features.drop(columns=timestamp_columns, errors='ignore')

    non_numeric_columns = features.select_dtypes(include=['object']).columns
    encoder = LabelEncoder()
    for col in non_numeric_columns:
        features[col] = encoder.fit_transform(features[col].astype(str))

    allowed_columns = ['SYMBOL', 'CREATED_BLOCK_TIMESTAMP', 'CREATOR_ADDRESS', 
                       'TO_ADDRESS_x', 'tx_count', 'TO_ADDRESS_y', 'tx_value_sum', 
                       'CONTRACT_ADDRESS_x', 'transfer_count', 'CONTRACT_ADDRESS_y', 
                       'transfer_value_sum']

    if is_training:
        allowed_columns.append('LABEL')

    if not is_training:
        allowed_columns.append('ADDRESS')  # Keep 'ADDRESS' for the test set

    existing_columns = [col for col in allowed_columns if col in features.columns]

    for col in allowed_columns:
        if col not in features.columns:
            features[col] = 0

    features = features[existing_columns]
    return features


features = preprocess_data(training_tokens, transactions, token_transfers, nft_transfers, dex_swaps, is_training=True)
X = features.drop(columns=['LABEL', 'ADDRESS'], errors='ignore')
y = features['LABEL']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

y_val_pred = rf_model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

test_features = preprocess_data(test_tokens, transactions, token_transfers, nft_transfers, dex_swaps, is_training=False)
X_test = test_features.drop(columns=['ADDRESS'], errors='ignore')
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)  # Ensure consistent columns

test_predictions = rf_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'ADDRESS': test_features['ADDRESS'],  # Ensure 'ADDRESS' exists
    'PRED': test_predictions
})
submission.to_csv("submission.csv", index=False)
