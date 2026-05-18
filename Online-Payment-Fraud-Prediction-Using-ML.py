# =========================
# 1. Import
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)

# Fixed to use set_theme which is the modern seaborn method
sns.set_theme(style="whitegrid")


# =========================
# 2. LOAD DATASET
# =========================
try:
    # Changed path from '/content/...' (Colab) to relative path (Local)
    # The CSV should be placed in the exact same folder as this script.
    data = pd.read_csv("fraud_legit_10k_30_70_ratio.csv")
    print("Dataset Loaded Successfully")
    print("Dataset Shape:", data.shape)
except FileNotFoundError:
    print("Error: Could not find 'fraud_legit_10k_30_70_ratio.csv'. Please ensure it's in the same folder as this script.")
    exit(1)


# =========================
# 3. DATA CLEANING (DETAILED REPORT)
# =========================

print("\n===== DATA CLEANING REPORT =====")

original_shape = data.shape
print(f"Original Dataset Shape: {original_shape}")

# 3.1 Drop unnecessary columns
columns_dropped = ['nameOrig', 'nameDest']
data.drop(columns_dropped, axis=1, inplace=True)
print(f"Columns Dropped: {columns_dropped}")

# 3.2 Missing Values
missing_before = data.isnull().sum().sum()
data.dropna(inplace=True)
missing_after = data.isnull().sum().sum()
print(f"Missing Values Removed: {missing_before - missing_after}")

# 3.3 Negative Amount Check
negative_amounts = data[data['amount'] < 0].shape[0]
data = data[data['amount'] >= 0]
print(f"Negative Amount Transactions Removed: {negative_amounts}")

# 3.4 Outlier Detection using IQR (for amount)
Q1 = data['amount'].quantile(0.25)
Q3 = data['amount'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['amount'] < lower_bound) | (data['amount'] > upper_bound)]
outlier_count = outliers.shape[0]

print(f"Outliers Detected in 'amount' column (IQR method): {outlier_count}")

# OPTIONAL: Remove outliers (uncomment if you want)
# data = data[(data['amount'] >= lower_bound) & (data['amount'] <= upper_bound)]
# print(f"Outliers Removed: {outlier_count}")

print(f"Final Dataset Shape After Cleaning: {data.shape}")
print("Data Cleaning Completed Successfully ✅")


# =========================
# 4. FEATURE ENGINEERING
# =========================
print("\n===== FEATURE ENGINEERING =====")

# Original Features
data['sender_balance_error'] = (
    data['oldbalanceOrg'] - data['amount'] - data['newbalanceOrig']
)

data['receiver_balance_error'] = (
    data['newbalanceDest'] - data['oldbalanceDest'] - data['amount']
)

# --- NEW FEATURES ADDED ---

# 1. Hour of the day (since 'step' represents hours, modulo 24 gives the exact hour of the day)
data['hour_of_day'] = data['step'] % 24

# 2. Amount to Balance Ratio (How much of the sender's balance is being moved?)
data['amount_to_balance_ratio'] = data['amount'] / (data['oldbalanceOrg'] + 1e-6)

# 3. Empties Account Flag (Does the transaction exactly empty the sender's account?)
data['empties_account'] = (data['amount'] == data['oldbalanceOrg']).astype(int)

# 4. Large Transaction Flag (Commonly, unusually large transfers are suspicious)
data['is_large_transaction'] = (data['amount'] > 200000).astype(int)

print("Features added successfully:")
print(" 1. sender_balance_error: Discrepancy in sender's balance")
print(" 2. receiver_balance_error: Discrepancy in receiver's balance")
print(" 3. hour_of_day: Exact hour of the transaction")
print(" 4. amount_to_balance_ratio: Proportion of sender's balance transferred")
print(" 5. empties_account: Flag if the transaction perfectly empties the account")
print(" 6. is_large_transaction: Flag for unusually large amounts (> 200,000)")

# =========================
# 5. ENCODE CATEGORICAL DATA
# =========================

le = LabelEncoder()
data['type'] = le.fit_transform(data['type'])


# =========================
# 6. ADVANCED DATA VISUALIZATION (EDA)
# =========================

print("\nGenerating Advanced Visualizations...")

plt.rcParams.update({'font.size': 11})

# ... (Visualizations are kept the same but simplified a bit for execution speed if needed, 
# but per request, leaving them identical to the original prompt)

# A. FRAUD VS LEGIT DISTRIBUTION
fraud_counts = data['isFraud'].value_counts()
plt.figure(figsize=(7,4))
colors = ['#2E7D32', '#C62828']
sns.barplot(x=fraud_counts.values, y=['Legit','Fraud'], palette=colors)
plt.title("Fraud vs Legit Transaction Distribution", fontsize=14, weight='bold')
plt.xlabel("Number of Transactions")
plt.ylabel("Transaction Category")
for index, value in enumerate(fraud_counts.values):
    plt.text(value, index, f'  {value:,}', va='center')
plt.tight_layout()
plt.show()

# B. FRAUD PERCENTAGE VISUAL 
fraud_percentage = (fraud_counts.get(1, 0) / len(data)) * 100
plt.figure(figsize=(6,4))
plt.bar(['Fraud Transactions'], [fraud_percentage], color='#C62828')
plt.title("Fraud Percentage in Dataset", fontsize=14, weight='bold')
plt.ylabel("Percentage (%)")
plt.text(0, fraud_percentage, f"{fraud_percentage:.3f}%", ha='center', va='bottom', fontsize=12, weight='bold')
plt.ylim(0, max(1, fraud_percentage * 1.5))
plt.tight_layout()
plt.show()

# C. TRANSACTION TYPE DISTRIBUTION
plt.figure(figsize=(8,5))
type_names = le.inverse_transform(data['type'])
type_counts = pd.Series(type_names).value_counts().sort_values()
sns.barplot(x=type_counts.values, y=type_counts.index, palette="viridis")
plt.title("Transaction Type Distribution", fontsize=14, weight='bold')
plt.xlabel("Count")
plt.ylabel("Transaction Type")
plt.tight_layout()
plt.show()

# D. HISTOGRAM – AMOUNT DISTRIBUTION (FRAUD VS LEGIT)
plt.figure(figsize=(8,5))
sns.histplot(data=data, x='amount', hue='isFraud', bins=50, log_scale=True, palette=['#2E7D32','#C62828'], alpha=0.6)
plt.title("Transaction Amount Distribution (Log Scale)", fontsize=14, weight='bold')
plt.xlabel("Transaction Amount (Log Scale)")
plt.ylabel("Frequency")
plt.legend(title="Fraud Status", labels=['Legit','Fraud'])
plt.tight_layout()
plt.show()

# E. SCATTER PLOT – AMOUNT vs SENDER BALANCE
plt.figure(figsize=(8,5))
sns.scatterplot(data=data, x='oldbalanceOrg', y='amount', hue='isFraud', palette=['#2E7D32','#C62828'], alpha=0.5)
plt.title("Scatter Plot: Sender Balance vs Amount", fontsize=14, weight='bold')
plt.xlabel("Sender Old Balance")
plt.ylabel("Transaction Amount")
plt.legend(title="Fraud Status", labels=['Legit','Fraud'])
plt.tight_layout()
plt.show()

# F. BOXPLOT – AMOUNT COMPARISON
plt.figure(figsize=(7,4))
sns.boxplot(x='isFraud', y='amount', data=data, palette=['#2E7D32','#C62828'])
plt.yscale('log')
plt.title("Transaction Amount Comparison (Log Scale)", fontsize=14, weight='bold')
plt.xlabel("Fraud Status")
plt.ylabel("Transaction Amount (Log Scale)")
plt.xticks([0,1], ['Legit','Fraud'])
plt.tight_layout()
plt.show()

# 7. CORRELATION HEATMAP
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), cmap='coolwarm', annot=False, linewidths=0.5, linecolor='gray')
plt.title("Feature Correlation Heatmap", fontsize=15, weight='bold')
plt.tight_layout()
plt.show()

# -------------------------------------------------
# H. PCA (PRINCIPAL COMPONENT ANALYSIS) 2D VISUALIZATION
# -------------------------------------------------
print("\nGenerating PCA 2D Visualization...")
# We use PCA purely to visualize the high-dimensional data in 2D space.
# We scale the features first since PCA is sensitive to variance.
X_temp = data.drop('isFraud', axis=1)
y_temp = data['isFraud']

pca_scaler = StandardScaler()
X_temp_scaled = pca_scaler.fit_transform(X_temp)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_temp_scaled)

pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['isFraud'] = y_temp.values

plt.figure(figsize=(8,6))
sns.scatterplot(
    x='Principal Component 1', 
    y='Principal Component 2', 
    hue='isFraud',
    palette=['#2E7D32','#C62828'], 
    data=pca_df, 
    alpha=0.6
)
plt.title("2D PCA Visualization of Transactions", fontsize=14, weight='bold')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Fraud Status", labels=['Legit', 'Fraud'])
plt.tight_layout()
plt.show()
print("Description: This scatter plot uses PCA to squash all numerical features into 2 dimensions to visualize clusters of Fraud vs Legit transactions.")


# =========================
# 7. FEATURE & TARGET SPLIT
# =========================
X = data.drop('isFraud', axis=1)
y = data['isFraud']


# =========================
# 8. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# =========================
# 9. FEATURE SCALING
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# 10. RANDOM FOREST MODEL
# =========================
rf = RandomForestClassifier(
    n_estimators=400,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced_subsample'
)

rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)


# =========================
# 11. MODEL EVALUATION
# =========================
print("\n===== RANDOM FOREST RESULTS =====")

# Accuracy
print("Accuracy:", accuracy_score(y_test, rf_pred))

# Confusion Matrix (Numeric)
cm = confusion_matrix(y_test, rf_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
print("Classification Report:\n", classification_report(y_test, rf_pred))


# =========================
# 12. CROSS FOLD VALIDATION (NEW)
# =========================
print("\n===== CROSS FOLD VALIDATION =====")
print("Running 5-Fold Stratified Cross-Validation on the training set...")

# StratifiedKFold preserves the proportion of legit vs fraud in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

print(f"Scores for each fold: {cv_scores}")
print(f"Mean Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")


# =========================
# 13. CONFUSION MATRIX HEATMAP
# =========================
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legit','Fraud'],
            yticklabels=['Legit','Fraud'])

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# =========================
# 14. ROC CURVE + AUC SCORE
# =========================
y_prob = rf.predict_proba(X_test_scaled)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

print("AUC Score:", auc_score)


# =========================
# 15. USER INPUT PREDICTION
# =========================
print("\n--- ENTER TRANSACTION DETAILS ---")

try:
    step = int(input("Step (hour): "))
    transaction_type = input("Transaction Type (CASH_OUT / TRANSFER): ").upper()
    amount = float(input("Amount: "))
    oldbalanceOrg = float(input("Sender Old Balance: "))
    newbalanceOrig = float(input("Sender New Balance: "))
    oldbalanceDest = float(input("Receiver Old Balance: "))
    newbalanceDest = float(input("Receiver New Balance: "))
    
    # Check if isFlaggedFraud is in training data; default to 0 if we need it
    isFlaggedFraud = 0

    type_encoded = le.transform([transaction_type])[0]

    # Original engineered features
    sender_error = oldbalanceOrg - amount - newbalanceOrig
    receiver_error = newbalanceDest - oldbalanceDest - amount

    # New engineered features calculation
    hour_of_day = step % 24
    amount_to_balance_ratio = amount / (oldbalanceOrg + 1e-6)
    empties_account = int(amount == oldbalanceOrg)
    is_large_transaction = int(amount > 200000)

    input_df = pd.DataFrame([{
        'step': step,
        'type': type_encoded,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'isFlaggedFraud': isFlaggedFraud,
        'sender_balance_error': sender_error,
        'receiver_balance_error': receiver_error,
        'hour_of_day': hour_of_day,
        'amount_to_balance_ratio': amount_to_balance_ratio,
        'empties_account': empties_account,
        'is_large_transaction': is_large_transaction
    }])

    # Reorder columns to match exactly what the model was trained on
    input_df = input_df[X.columns]

    input_scaled = scaler.transform(input_df)

    # 1. Get ML probability
    fraud_probability = rf.predict_proba(input_scaled)[0][1]
    
    # 2. Apply Rule-Based Anomaly Detection (Industry Standard Hybrid Approach)
    # The training dataset mostly features fraud where the account is perfectly emptied.
    # To catch logical anomalies (like massive missing funds), we use a rule overlay
    # that scales dynamically based on how large the discrepancy is.
    is_anomaly = False
    
    if abs(sender_error) > 5000:
        # Calculate severity (caps at 1.0 for missing funds >= 100,000)
        anomaly_severity = min(abs(sender_error) / 100000, 1.0)
        # Probability scales dynamically from 85% up to 99%
        dynamic_boost = 0.85 + (0.14 * anomaly_severity)
        fraud_probability = max(fraud_probability, dynamic_boost)
        is_anomaly = True
        
    elif abs(receiver_error) > 5000 and transaction_type == 'TRANSFER':
         anomaly_severity = min(abs(receiver_error) / 100000, 1.0)
         # Probability scales dynamically from 75% up to 95%
         dynamic_boost = 0.75 + (0.20 * anomaly_severity)
         fraud_probability = max(fraud_probability, dynamic_boost)
         is_anomaly = True

    threshold = 0.35
    prediction = 1 if fraud_probability > threshold else 0


    # =========================
    # 16. RESULT OUTPUT + EXPLANATION
    # =========================
    print("\n===== PREDICTION RESULT =====")

    if prediction == 1:
        print("\n❌ FRAUDULENT TRANSACTION DETECTED")
    else:
        print("\n✅ LEGIT TRANSACTION")

    print(f"Fraud Risk Probability: {fraud_probability*100:.2f}%")

    print("\n--- PREDICTION EXPLANATION ---")

    # 1. Common Transaction Type
    most_common_type_encoded = data['type'].mode()[0]
    most_common_type = le.inverse_transform([most_common_type_encoded])[0]
    print(f"1. Most common transaction type in dataset: {most_common_type}")

    # 2. Overall fraud risk level interpretation
    if fraud_probability < 0.20:
        risk_level = "Very Low"
    elif fraud_probability < 0.40:
        risk_level = "Low"
    elif fraud_probability < 0.70:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    print(f"2. Overall fraud risk level: {risk_level}")

    # Additional intelligent explanation
    if prediction == 1:
        if is_anomaly:
            print("Reason: MASSIVE BALANCE ANOMALY. The sender/receiver balances do not mathematically align with the transaction amount, which strongly indicates manipulation or hidden system fraud.")
        else:
            print("Reason: Transaction pattern resembles previously detected fraudulent behavior in the historical dataset.")
    else:
        print("Reason: Transaction pattern aligns with majority of legitimate transactions.")

except Exception as e:
    print(f"\nUser input interrupted or invalid: {e}")

print("\nMODEL EXECUTION COMPLETED SUCCESSFULLY ✅")