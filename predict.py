import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss

df = pd.read_csv('BTC-2017min.csv')



df.rename(columns={
    'date': 'Timestamp',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'Volume BTC': 'Volume_BTC',
    'Volume USD': 'Volume_USD'
}, inplace=True)

# Sort by Timestamp
df = df.sort_values('Timestamp').reset_index(drop=True)

# # Basic Info
# print(df.info())
# print(df.head())

df['Future_Close'] = df['Close'].shift(-5)
df['Future_Return'] = (df['Future_Close'] - df['Close'])/df['Close']

#bins and labels
bins = [-np.inf, -0.002, -0.001, 0, 0.001, 0.002, np.inf]
labels = [0, 1, 2, 3, 4, 5]

# Target class => range -> label
df['Target_Class'] = pd.cut(df['Future_Return'], bins=bins, labels=labels)

# drop last 5 rows with NaN in Future_Close
df = df.dropna(subset=['Target_Class']).reset_index(drop=True)

# set to int
df['Target_Class'] = df['Target_Class'].astype('int')

# print(df['Target_Class'].value_counts())
# print((df['Target_Class']==2).count())

df['Close_Mean_5'] = df['Close'].rolling(window=5).mean()
df['Close_Mean_10'] = df['Close'].rolling(window=10).mean()
df['Close_Mean_30'] = df['Close'].rolling(window=30).mean()

df['Close_Std_5'] = df['Close'].rolling(window=5).std()
df['Close_Std_10'] = df['Close'].rolling(window=10).std()
df['Close_Std_30'] = df['Close'].rolling(window=30).std()

df['High_Low_Range_5'] = df['High'].rolling(window=5).max() - df['Low'].rolling(window=5).min()
df['High_Low_Range_10'] = df['High'].rolling(window=10).max() - df['Low'].rolling(window=10).min()
df['High_Low_Range_30'] = df['High'].rolling(window=30).max() - df['Low'].rolling(window=30).min()

df['Close_Skew_5'] = df['Close'].rolling(window=5).skew()
df['Close_Kurt_5'] = df['Close'].rolling(window=5).kurt()
df['Close_Skew_10'] = df['Close'].rolling(window=10).skew()
df['Close_Kurt_10'] = df['Close'].rolling(window=10).skew()

df = df.dropna().reset_index(drop=True)

df['LogRet_1'] = np.log(df['Close'] / df['Close'].shift(1))
df['LogRet_5'] = np.log(df['Close'] / df['Close'].shift(5))
df['LogRet_10'] = np.log(df['Close'] / df['Close'].shift(10))
df['LogRet_30'] = np.log(df['Close'] / df['Close'].shift(30))

df['ROC_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
df['ROC_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)

df['Range_Pos_5'] = (df['Close'] - df['Low'].rolling(5).min()) / (df['High'].rolling(5).max() - df['Low'].rolling(5).min())
df['Range_Pos_10'] = (df['Close'] - df['Low'].rolling(10).min()) / (df['High'].rolling(10).max() - df['Low'].rolling(10).min())

df['Price_Slope_5'] = df['Close'].diff(4) / 4  # Approximate slope over 5 points
df['Price_Slope_10'] = df['Close'].diff(9) / 9


df = df.dropna().reset_index(drop=True)

df['Volatility_Close_5'] = df['Close'].rolling(window=5).std()
df['Volatility_Close_10'] = df['Close'].rolling(window=10).std()

df['LogRet_1'] = np.log(df['Close'] / df['Close'].shift(1))  # in case not already computed
df['Volatility_LogRet_5'] = df['LogRet_1'].rolling(window=5).std()
df['Volatility_LogRet_10'] = df['LogRet_1'].rolling(window=10).std()

# True Range: max(high-low, abs(high-prev_close), abs(low-prev_close))
high = df['High']
low = df['Low']
prev_close = df['Close'].shift(1)

df['True_Range'] = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))

# ATR: rolling average of True Range
df['ATR_5'] = df['True_Range'].rolling(window=5).mean()
df['ATR_10'] = df['True_Range'].rolling(window=10).mean()


df['ZScore_LogRet_5'] = (df['LogRet_1'] - df['LogRet_1'].rolling(5).mean()) / df['LogRet_1'].rolling(5).std()

df = df.dropna().reset_index(drop=True)



df['Volume_5'] = df['Volume_BTC'].rolling(window=5).sum()
df['Volume_10'] = df['Volume_BTC'].rolling(window=10).sum()


df['Rel_Volume_5'] = df['Volume_BTC'] / df['Volume_BTC'].rolling(window=5).mean()
df['Rel_Volume_10'] = df['Volume_BTC'] / df['Volume_BTC'].rolling(window=10).mean()


df['Price_Volume_Ratio_5'] = (df['Close'] - df['Close'].shift(5)) / df['Volume_5']
df['Price_Volume_Ratio_10'] = (df['Close'] - df['Close'].shift(10)) / df['Volume_10']


df['Volume_Change_1'] = df['Volume_BTC'].pct_change()


df = df.dropna().reset_index(drop=True)


# Candle Body Size
df['Candle_Body'] = abs(df['Close'] - df['Open'])

# Upper and Lower Shadows
df['Upper_Shadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
df['Lower_Shadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']

# Body-to-Range Ratio
df['Candle_Range'] = df['High'] - df['Low']
df['Body_Ratio'] = df['Candle_Body'] / df['Candle_Range'].replace(0, np.nan)



# Rolling mean and max/min
df['Rolling_Mean_10'] = df['Close'].rolling(window=10).mean()
df['Rolling_Max_10'] = df['Close'].rolling(window=10).max()
df['Rolling_Min_10'] = df['Close'].rolling(window=10).min()

# Price position in rolling range (0 = low, 1 = high)
df['Price_Position_10'] = (df['Close'] - df['Rolling_Min_10']) / (df['Rolling_Max_10'] - df['Rolling_Min_10']).replace(0, np.nan)


df['Close/Open'] = df['Close'] / df['Open']
df['High/Low'] = df['High'] / df['Low']
df['Close/High'] = df['Close'] / df['High']
df['Close/Low'] = df['Close'] / df['Low']


df = df.dropna().reset_index(drop=True)


# Drop columns you won't use in modeling
drop_cols = ['Timestamp', 'Future_Close', 'Future_Return', 'symbol']  # any target leakage columns
features = [col for col in df.columns if col not in drop_cols + ['Target_Class']]

X = df[features]
y = df['Target_Class']

X = X.replace([np.inf, -np.inf], np.nan)

# Drop rows with any NaNs (inf got converted to NaN above)
X = X.dropna()
y = y.loc[X.index]

# print(features)


split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Fit on training data
clf.fit(X_train_scaled, y_train)

# Predict class labels
y_pred = clf.predict(X_test_scaled)

# Predict class probabilities
y_proba = clf.predict_proba(X_test_scaled)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


plt.figure(figsize=(12, 6))
for class_idx in range(y_proba.shape[1]):
    plt.plot(y_proba[:, class_idx], label=f'Class {class_idx}')

plt.title("Predicted Class Probabilities Over Time (Test Set)")
plt.xlabel("Time Step (Test Index)")
plt.ylabel("Probability")
plt.legend()
plt.tight_layout()
plt.show()
