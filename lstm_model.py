import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Masking

# Load dataset
df = pd.read_csv("fraud_flagged_transactions.csv") 
print(f"Dataset loaded: {df.shape} rows, {df.columns.tolist()}")

# Sort transactions chronologically by account
df = df.sort_values(['AccountID', 'TransactionDate'])

# Encode categorical features
categorical_cols = ['DeviceID', 'Location', 'TransactionType', 'Channel']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Scale numerical features
numeric_cols = ['TransactionAmount', 'CustomerAge', 'TransactionDuration',
                'LoginAttempts', 'AccountBalance', 'TimeSinceLastTxn', 'TxnCount2Min']
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Final features and target
feature_cols = numeric_cols + categorical_cols
target_col = 'Fraud_Label'

# Split data into train and test sets by account
accounts = df['AccountID'].unique()
train_accounts, test_accounts = train_test_split(accounts, test_size=0.2, random_state=42)
train_df = df[df['AccountID'].isin(train_accounts)]
test_df = df[df['AccountID'].isin(test_accounts)]

# Prepare sequences for training
grouped_train = train_df.groupby('AccountID')
X_train, y_train = [], []

for _, group in grouped_train:
    X_train.append(group[feature_cols].values)
    y_train.append(group[target_col].values)

max_seq_len = 50
X_train_padded = pad_sequences(X_train, maxlen=max_seq_len, dtype='float32', padding='pre', truncating='pre')
y_train_padded = pad_sequences(y_train, maxlen=max_seq_len, dtype='int32', padding='pre', truncating='pre')
y_train_padded = np.expand_dims(y_train_padded, axis=-1)

print(f"Train X padded shape: {X_train_padded.shape}")
print(f"Train y padded shape: {y_train_padded.shape}")

# Prepare sequences for testing
grouped_test = test_df.groupby('AccountID')
X_test, y_test = [], []

for _, group in grouped_test:
    X_test.append(group[feature_cols].values)
    y_test.append(group[target_col].values)

X_test_padded = pad_sequences(X_test, maxlen=max_seq_len, dtype='float32', padding='pre', truncating='pre')
y_test_padded = pad_sequences(y_test, maxlen=max_seq_len, dtype='int32', padding='pre', truncating='pre')
y_test_padded = np.expand_dims(y_test_padded, axis=-1)

print(f"Test X padded shape: {X_test_padded.shape}")
print(f"Test y padded shape: {y_test_padded.shape}")

# Flatten labels to compute class weights
flat_labels = y_train_padded.flatten()
nonzero_mask = flat_labels != 0
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(flat_labels[nonzero_mask]), y=flat_labels[nonzero_mask])
class_weights_dict = dict(enumerate(class_weights))
print(f"Computed class weights: {class_weights_dict}")

# Build model
model = Sequential([
    Masking(mask_value=0., input_shape=(max_seq_len, len(feature_cols))),
    LSTM(64, return_sequences=True),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(X_train_padded, y_train_padded, 
                    epochs=10, 
                    batch_size=32, 
                    validation_split=0.2,
                    class_weight=class_weights_dict)

# Evaluate model
loss, accuracy = model.evaluate(X_test_padded, y_test_padded)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predict on test data
y_pred_prob = model.predict(X_test_padded)
y_pred = (y_pred_prob > 0.2).astype(int)

# Mask padded values
flat_y_true = y_test_padded.flatten()
flat_y_pred = y_pred.flatten()
flat_y_prob = y_pred_prob.flatten()
mask = flat_y_true != 0

# Apply mask
y_true_masked = flat_y_true[mask]
y_pred_masked = flat_y_pred[mask]
y_prob_masked = flat_y_prob[mask]

# Show example predictions
for i in range(min(10, len(X_test_padded))):
    print(f"True label: {y_test_padded[i].flatten()}")
    print(f"Predicted prob: {y_pred_prob[i].flatten()}")
    print(f"Predicted label: {y_pred[i].flatten()}")
    print("---")

# Print classification report and confusion matrix
print("\nClassification Report (masked):")
print(classification_report(y_true_masked, y_pred_masked, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_true_masked, y_pred_masked))

# Plot Precision-Recall vs Threshold
precision, recall, thresholds = precision_recall_curve(y_true_masked, y_prob_masked)
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.title('Precision and Recall vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid()
plt.show()

# Save model
model.save("lstm_fraud_model.h5")
print("Model saved as 'lstm_fraud_model.h5'")
