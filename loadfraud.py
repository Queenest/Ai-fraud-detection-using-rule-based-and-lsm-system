import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Masking

# Load the dataset
file_path = "bank_transactions_data_2.csv"  # Replace with your file name if different
df = pd.read_csv("bank_transactions_data_2 (1).csv")

# --- Cleaning Steps --- 

# 1. Convert date columns to datetime format
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'], errors='coerce')

# 2. Sort transactions by AccountID and TransactionDate
df.sort_values(by=['AccountID', 'TransactionDate'], inplace=True)

# 3. Reset index after sorting
df.reset_index(drop=True, inplace=True)

# 4. Optional: check for and report missing values
missing = df.isnull().sum()
print("Missing values per column:\n", missing)

# 5. Preview the cleaned dataset
print("\nCleaned Dataset Preview:\n", df.head())

# Save cleaned dataset (optional)
df.to_csv("cleaned_bank_transactions.csv", index=False)

# Load cleaned data (or continue from previous steps)
df = pd.read_csv("cleaned_bank_transactions.csv")

# Sort for consistency
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
df.sort_values(by=['AccountID', 'TransactionDate'], inplace=True)

# RULE 1: High Amount Flag
# Create empty column for R1 flag
df['R1_HighAmount'] = 0

# Define function to compute high amount flag
def flag_high_amount(group):
    amt = group['TransactionAmount']
    r1 = pd.Series(0, index=group.index)
    
    if len(group) > 5:
        threshold = amt.expanding().quantile(0.95).shift(1)  # use 95th percentile
        r1 = (amt > threshold).astype(int)
    
    group = group.copy()
    group['R1_HighAmount'] = r1
    return group

# Apply per account — clean version, no deprecated behavior
df_temp = df.groupby('AccountID', group_keys=False)[
    ['TransactionID', 'TransactionAmount', 'TransactionDate', 'R1_HighAmount']
].apply(flag_high_amount)

# Merge back with original dataframe
df = df_temp.merge(df.drop(columns=['R1_HighAmount']), on=['TransactionID'], suffixes=('', '_old'))

# Drop any duplicate columns
df = df.loc[:, ~df.columns.str.endswith('_old')]

# Preview result
print("Rule 1 - High Amount Flag:")
print(df[['AccountID', 'TransactionAmount', 'R1_HighAmount']].head(10))

# RULE 2: High Login Attempts Flag
median_logins = df.groupby('AccountID')['LoginAttempts'].transform('median')
df['R2_HighLoginAttempts'] = (df['LoginAttempts'] > median_logins * 1.5).astype(int)

# Preview the result
print("\nRule 2 - High Login Attempts:")
print(df[['LoginAttempts', 'R2_HighLoginAttempts']].head(10))

# RULE 3: Very Frequent Transactions Flag
# Compute PreviousTransactionDate *after sorting*
df.sort_values(by=['AccountID', 'TransactionDate'], inplace=True)
df['PreviousTransactionDate_calc'] = df.groupby('AccountID')['TransactionDate'].shift(1)
df['TimeSinceLastTxn'] = (df['TransactionDate'] - df['PreviousTransactionDate_calc']).dt.total_seconds()
df['TimeSinceLastTxn'] = df['TimeSinceLastTxn'].fillna(-1)

def create_frequency_flags(df):
    df = df.sort_values(['AccountID', 'TransactionDate'])
    df['TxnCount2Min'] = 0
    
    for account_id in df['AccountID'].unique():
        mask = df['AccountID'] == account_id
        account_data = df[mask].copy()
        
        # Set TransactionDate as index for rolling window calculation
        account_data = account_data.set_index('TransactionDate')
        account_data['TxnCount2Min'] = account_data['TransactionID'].rolling('2min').count()
        
        # Update the main dataframe
        df.loc[mask, 'TxnCount2Min'] = account_data['TxnCount2Min'].values
    
    return df

# Apply frequency analysis without groupby apply
df = create_frequency_flags(df)
df['R3_VeryFrequent'] = (df['TxnCount2Min'] >= 3).astype(int)

# Preview
print("\nRule 3 - Very Frequent Transactions:")
print(df[['AccountID', 'TransactionDate', 'TimeSinceLastTxn', 'R3_VeryFrequent']].head(10))


# RULE 4: Suspicious Device Flag
def create_device_flags(df):
    df = df.sort_values(['AccountID', 'TransactionDate'])
    df['R4_SuspiciousDevice'] = 0
    
    for account_id in df['AccountID'].unique():
        mask = df['AccountID'] == account_id
        account_data = df[mask]
        
        device_counts = {}
        flags = []
        
        for i, dev in enumerate(account_data['DeviceID']):
            # First 10 transactions are always legitimate
            if i < 10:
                flags.append(0)
            else:
                # Count how many times we've seen this device
                device_counts[dev] = device_counts.get(dev, 0) + 1
                
                # If device appears 2+ times, it's legitimate (0)
                # If it's new (first time), it's suspicious (1)
                if device_counts[dev] >= 5:
                    flags.append(0)  # Legitimate
                else:
                    flags.append(1)  # Suspicious (new device)
        
        df.loc[mask, 'R4_SuspiciousDevice'] = flags
    
    return df

# RULE 5: Suspicious Location Flag
def create_location_flags(df):
    df = df.sort_values(['AccountID', 'TransactionDate'])
    df['R5_SuspiciousLocation'] = 0
    
    for account_id in df['AccountID'].unique():
        mask = df['AccountID'] == account_id
        account_data = df[mask]
        
        location_counts = {}
        flags = []
        
        for i, loc in enumerate(account_data['Location']):
            # First 10 transactions are always legitimate  
            if i < 10:
                flags.append(0)
            else:
                # Count how many times we've seen this location
                location_counts[loc] = location_counts.get(loc, 0) + 1
                
                # If location appears 2+ times, it's legitimate (0)
                # If it's new (first time), it's suspicious (1)
                if location_counts[loc] >= 5:
                    flags.append(0)  # Legitimate
                else:
                    flags.append(1)  # Suspicious (new location)
        
        df.loc[mask, 'R5_SuspiciousLocation'] = flags
    
    return df

# Apply rules - FIXED: Use direct iteration instead of groupby apply
print("Before grouping for device/location check:", df.columns.tolist())

# Sort for proper ordering within group
df = df.sort_values(['AccountID', 'TransactionDate'])

# Apply the functions using direct iteration (avoids pandas groupby issues)
df = create_device_flags(df)
df = create_location_flags(df)

print("\nRule 4 & 5 - Device and Location Flags:")
print(df[['AccountID', 'DeviceID', 'Location', 'R4_SuspiciousDevice', 'R5_SuspiciousLocation']].head(10))

# RULE 6: Composite Fraud Rule - High Amount + (Suspicious Device OR Suspicious Location)
df['R6_HighAmountSuspiciousContext'] = ((df['R1_HighAmount'] == 1) & 
                                         ((df['R4_SuspiciousDevice'] == 1) | (df['R5_SuspiciousLocation'] == 1))).astype(int)

print("\nRule 6 - Composite Fraud Detection (High Amount + Suspicious Context):")
print(df[['AccountID', 'TransactionAmount', 'R1_HighAmount', 'R4_SuspiciousDevice', 'R5_SuspiciousLocation', 'R6_HighAmountSuspiciousContext']].head(10))


# RULE 7: Suspicious Merchant Flag
# Calculate fraud rate per MerchantID using Rule 6 as the fraud indicator
merchant_fraud_rate = df.groupby('MerchantID')['R6_HighAmountSuspiciousContext'].mean()

# Identify suspicious merchants (fraud rate > 0.5)
suspicious_merchants = merchant_fraud_rate[merchant_fraud_rate > 0.5].index

# Flag transactions with suspicious merchants
df['R7_SuspiciousMerchant'] = df['MerchantID'].isin(suspicious_merchants).astype(int)

print(f"\nRule 7 - Suspicious Merchant Analysis:")
print(f"Merchants with fraud rate > 0.5: {len(suspicious_merchants)}")
print(f"Merchant fraud rates (top 10):")
print(merchant_fraud_rate.sort_values(ascending=False).head(10))

print(f"\nRule 7 - Suspicious Merchant Flags:")
print(df[['MerchantID', 'R6_HighAmountSuspiciousContext', 'R7_SuspiciousMerchant']].head(10))

# RULE 8: Composite Rule - Flag if Rule 6 OR Suspicious Merchant
df['R8_CompositeRule'] = ((df['R6_HighAmountSuspiciousContext'] == 1) | 
                          (df['R7_SuspiciousMerchant'] == 1)).astype(int)

print(f"\nRule 8 - Final Composite Rule (Rule 6 OR Suspicious Merchant):")
print(df[['AccountID', 'TransactionAmount', 'MerchantID', 'R6_HighAmountSuspiciousContext', 'R7_SuspiciousMerchant', 'R8_CompositeRule']].head(10))

# List of your rule columns
rule_columns = ['R1_HighAmount', 'R2_HighLoginAttempts', 'R3_VeryFrequent', 'R4_SuspiciousDevice', 'R5_SuspiciousLocation', 'R6_HighAmountSuspiciousContext']

# Create final fraud label combining all rules
df['Fraud_Label'] = (df[rule_columns].sum(axis=1) > 0).astype(int)

# Optional prints
print(df[['TransactionID'] + rule_columns + ['Fraud_Label']].head(15))
print("\nFraud label distribution:\n", df['Fraud_Label'].value_counts(normalize=True))

# Save the final dataset with all flags
df.to_csv("fraud_flagged_transactions.csv", index=False)
print("\nFinal dataset saved as 'fraud_flagged_transactions.csv'")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")







