import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_auc_score, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Masking, BatchNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Custom Focal Loss for imbalanced data
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        cross_entropy = -tf.math.log(p_t)
        return alpha_t * tf.pow(1 - p_t, gamma) * cross_entropy
    return focal_loss_fixed

# Load dataset with rule-based flags
file_path = "fraud_flagged_transactions.csv"
df = pd.read_csv(file_path)
print(f"Dataset loaded: {df.shape} rows")

# Convert date and sort
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
df = df.sort_values(['AccountID', 'TransactionDate'])

# Include rule-based columns R1 to R8
rule_cols = ['R1_HighAmount', 'R2_HighLoginAttempts', 'R3_VeryFrequent',
             'R4_SuspiciousDevice', 'R5_SuspiciousLocation',
             'R6_HighAmountSuspiciousContext', 'R7_SuspiciousMerchant', 'R8_CompositeRule']

feature_cols = rule_cols
target_col = 'Fraud_Label'

# Prepare transaction-level sequences for each row with context window
def prepare_transaction_level_sequences(data, max_seq_len=10):
    sequences = []
    labels = []

    for account_id, group in data.groupby('AccountID'):
        group = group.sort_values('TransactionDate')
        features = group[feature_cols].values
        targets = group[target_col].values

        for i in range(len(features)):
            start_idx = max(0, i - max_seq_len + 1)
            seq = features[start_idx:i+1]
            seq_padded = pad_sequences([seq], maxlen=max_seq_len, dtype='float32', padding='pre', value=-1.0)[0]
            sequences.append(seq_padded)
            labels.append(targets[i])

    return np.array(sequences), np.array(labels)

# Temporal split
cutoff_date = df['TransactionDate'].quantile(0.8)
print(f"Temporal cutoff date: {cutoff_date}")
train_df = df[df['TransactionDate'] <= cutoff_date]
test_df = df[df['TransactionDate'] > cutoff_date]

print(f"Temporal Train transactions: {len(train_df)}")
print(f"Temporal Test transactions: {len(test_df)}")

X_train, y_train = prepare_transaction_level_sequences(train_df)
X_test, y_test = prepare_transaction_level_sequences(test_df)

print(f"X_train shape: {X_train.shape}, y_train: {np.bincount(y_train)}")
print(f"X_test shape: {X_test.shape}, y_test: {np.bincount(y_test)}")

# Class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights (temporal):", class_weights_dict)

# Model definition
def create_improved_model(input_shape, use_focal_loss=True):
    model = Sequential([
        Masking(mask_value=-1., input_shape=input_shape),
        LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    loss_fn = focal_loss(gamma=2.0, alpha=0.25) if use_focal_loss else 'binary_crossentropy'
    model.compile(optimizer=Adam(learning_rate=0.001), loss=loss_fn, metrics=['accuracy', 'precision', 'recall'])
    return model

model = create_improved_model((X_train.shape[1], X_train.shape[2]), use_focal_loss=True)
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

# Train
history = model.fit(X_train, y_train, validation_split=0.2, class_weight=class_weights_dict,
                    batch_size=32, epochs=50, callbacks=callbacks, verbose=1)

# Evaluate
loss, acc, precision, recall = model.evaluate(X_test, y_test, verbose=0)
y_pred_prob = model.predict(X_test, verbose=0).flatten()

# Threshold selection
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_prob)
f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
y_pred_optimal = (y_pred_prob > optimal_threshold).astype(int)

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred_optimal)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Threshold = {optimal_threshold:.3f})")
plt.tight_layout()
plt.savefig("confusion_matrix_transaction_level.png", dpi=300)
plt.show()

# Heatmap: Predicted probabilities vs true labels
plt.figure(figsize=(10, 5))
sns.histplot(x=y_pred_prob, hue=y_test, bins=50, kde=True, palette=['blue', 'red'], stat='density')
plt.axvline(optimal_threshold, color='black', linestyle='--', label=f'Threshold = {optimal_threshold:.3f}')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Distribution of Predicted Probabilities by Class')
plt.legend()
plt.tight_layout()
plt.savefig("predicted_probabilities_distribution.png", dpi=300)
plt.show()

# Metrics summary
auc_roc = roc_auc_score(y_test, y_pred_prob)
pr_auc = auc(recall_vals, precision_vals)
print(f"\nAUC-ROC: {auc_roc:.4f}")
print(f"AUC-PR : {pr_auc:.4f}")
print(f"Optimal Threshold: {optimal_threshold:.3f}")
