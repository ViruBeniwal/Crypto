import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, log_loss

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from a CSV file, renames columns, and sorts by timestamp.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The prepared and sorted DataFrame.
    """
    print("Loading and preparing data...")
    df = pd.read_csv(filepath)
    df.rename(columns={
        'date': 'Timestamp',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'Volume BTC': 'Volume_BTC',
        'Volume USD': 'Volume_USD'
    }, inplace=True)
    df = df.sort_values('Timestamp').reset_index(drop=True)
    return df

def engineer_target_variable(df: pd.DataFrame, future_periods: int, bins: list, labels: list) -> pd.DataFrame:
    """
    Engineers the target variable for classification.

    Args:
        df (pd.DataFrame): The input DataFrame.
        future_periods (int): The number of periods to look into the future for the return.
        bins (list): The bin edges for categorizing the future return.
        labels (list): The numeric labels for each bin.

    Returns:
        pd.DataFrame: The DataFrame with 'Future_Return' and 'Target_Class' columns.
    """
    print("Engineering the target variable...")
    df['Future_Close'] = df['Close'].shift(-future_periods)
    df['Future_Return'] = (df['Future_Close'] - df['Close']) / df['Close']
    df['Target_Class'] = pd.cut(df['Future_Return'], bins=bins, labels=labels)
    df.dropna(subset=['Target_Class'], inplace=True)
    df['Target_Class'] = df['Target_Class'].astype('int')
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    A master function to engineer all features.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data.

    Returns:
        pd.DataFrame: DataFrame with all engineered features.
    """
    print("Engineering features...")
    # Price-Based Features
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
    df['Close_Kurt_10'] = df['Close'].rolling(window=10).kurt()

    # Return and Momentum Features
    df['LogRet_1'] = np.log(df['Close'] / df['Close'].shift(1))
    df['LogRet_5'] = np.log(df['Close'] / df['Close'].shift(5))
    df['LogRet_10'] = np.log(df['Close'] / df['Close'].shift(10))
    df['LogRet_30'] = np.log(df['Close'] / df['Close'].shift(30))
    df['ROC_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
    df['ROC_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)
    df['Range_Pos_5'] = (df['Close'] - df['Low'].rolling(5).min()) / (df['High'].rolling(5).max() - df['Low'].rolling(5).min())
    df['Range_Pos_10'] = (df['Close'] - df['Low'].rolling(10).min()) / (df['High'].rolling(10).max() - df['Low'].rolling(10).min())
    df['Price_Slope_5'] = df['Close'].diff(4) / 4
    df['Price_Slope_10'] = df['Close'].diff(9) / 9

    # Advanced Volatility Features
    df['Volatility_LogRet_5'] = df['LogRet_1'].rolling(window=5).std()
    df['Volatility_LogRet_10'] = df['LogRet_1'].rolling(window=10).std()
    high_low = df['High'] - df['Low']
    high_prev_close = abs(df['High'] - df['Close'].shift(1))
    low_prev_close = abs(df['Low'] - df['Close'].shift(1))
    df['True_Range'] = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))
    df['ATR_5'] = df['True_Range'].rolling(window=5).mean()
    df['ATR_10'] = df['True_Range'].rolling(window=10).mean()
    df['ZScore_LogRet_5'] = (df['LogRet_1'] - df['LogRet_1'].rolling(5).mean()) / df['LogRet_1'].rolling(5).std()

    # Volume-Based Features
    df['Volume_5'] = df['Volume_BTC'].rolling(window=5).sum()
    df['Volume_10'] = df['Volume_BTC'].rolling(window=10).sum()
    df['Rel_Volume_5'] = df['Volume_BTC'] / df['Volume_BTC'].rolling(window=5).mean()
    df['Rel_Volume_10'] = df['Volume_BTC'] / df['Volume_BTC'].rolling(window=10).mean()
    df['Price_Volume_Ratio_5'] = (df['Close'] - df['Close'].shift(5)) / df['Volume_5']
    df['Price_Volume_Ratio_10'] = (df['Close'] - df['Close'].shift(10)) / df['Volume_10']
    df['Volume_Change_1'] = df['Volume_BTC'].pct_change()

    # Candlestick-Based Features
    df['Candle_Body'] = abs(df['Close'] - df['Open'])
    df['Upper_Shadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['Lower_Shadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    df['Candle_Range'] = df['High'] - df['Low']
    df['Body_Ratio'] = df['Candle_Body'] / df['Candle_Range']

    # Price and Ratio Features
    df['Close/Open'] = df['Close'] / df['Open']
    df['High/Low'] = df['High'] / df['Low']
    df['Close/High'] = df['Close'] / df['High']
    df['Close/Low'] = df['Close'] / df['Low']
    
    return df

def clean_and_prepare_final_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame by handling infinities and dropping NaNs.

    Args:
        df (pd.DataFrame): The DataFrame with engineered features.

    Returns:
        pd.DataFrame: The cleaned, final DataFrame ready for modeling.
    """
    print("Cleaning final DataFrame...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def split_and_scale_data(df: pd.DataFrame, features: list, target_col: str, test_split_ratio: float):
    """
    Splits data into features (X) and target (y), then into train/test sets, and scales the features.

    Args:
        df (pd.DataFrame): The final, cleaned DataFrame.
        features (list): The list of feature column names.
        target_col (str): The name of the target column.
        test_split_ratio (float): The proportion of the dataset to allocate to the test set.

    Returns:
        tuple: A tuple containing X_train_scaled, X_test_scaled, y_train, y_test.
    """
    print("Splitting and scaling data...")
    X = df[features]
    y = df[target_col]

    split_idx = int(len(df) * (1 - test_split_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_and_evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test, range_labels: list):
    """
    Initializes, trains, and evaluates the Random Forest Classifier.

    Args:
        X_train_scaled: Scaled training features.
        y_train: Training target variable.
        X_test_scaled: Scaled testing features.
        y_test: Testing target variable.
        range_labels (list): Human-readable labels for the target classes.

    Returns:
        tuple: A tuple containing the trained classifier and the predicted probabilities on the test set.
    """
    print("Training the Random Forest model...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)
    print("Training complete.")

    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {100*accuracy_score(y_test, y_pred):.4f}")
    print(f"Log Loss: {log_loss(y_test, y_proba):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=range_labels))
    
    return clf, y_proba

def plot_results(y_test, y_proba, range_labels: list):
    """
    Plots the predicted probabilities for each class over time.

    Args:
        y_test: The true labels for the test set (used for index).
        y_proba: The predicted probabilities from the model.
        range_labels (list): The human-readable labels for the classes.
    """
    print("\nGenerating plot of predicted probabilities...")
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(15, 8))

    for class_idx in range(y_proba.shape[1]):
        plt.plot(y_test.index, y_proba[:, class_idx], label=f'Range: {range_labels[class_idx]}')

    plt.title("Predicted Probabilities for Each Return Range (Test Set)", fontsize=16)
    plt.xlabel("Time Step (Test Index)", fontsize=12)
    plt.ylabel("Predicted Probability", fontsize=12)
    plt.legend(title="Future Return Range (5 min)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    print("Plot displayed.")

def main():
    """
    Main function to run the entire ML pipeline.
    """
    # --- Configuration ---
    FILEPATH = 'BTC-2017min.csv'
    FUTURE_PERIODS = 5
    BINS = [-np.inf, -0.002, -0.001, 0, 0.001, 0.002, np.inf]
    RANGE_LABELS = [
        '<-0.2%', '-0.2% to -0.1%', '-0.1% to 0%',
        '0% to 0.1%', '0.1% to 0.2%', '>0.2%'
    ]
    TEST_SPLIT_RATIO = 0.2

    # --- Pipeline ---
    # 1. Load and Prepare
    df = load_and_prepare_data(FILEPATH)

    # 2. Engineer Target
    df = engineer_target_variable(df, FUTURE_PERIODS, BINS, range(len(RANGE_LABELS)))

    # 3. Engineer Features
    df = engineer_features(df)
    
    # 4. Clean and Finalize
    df = clean_and_prepare_final_df(df)

    # 5. Define Features and Split/Scale Data
    drop_cols = ['Timestamp', 'Future_Close', 'Future_Return', 'symbol', 'True_Range', 'Candle_Range']
    features = [col for col in df.columns if col not in drop_cols and col != 'Target_Class']
    
    X_train_scaled, X_test_scaled, y_train, y_test, _ = split_and_scale_data(
        df, features, 'Target_Class', TEST_SPLIT_RATIO
    )

    # 6. Train Model and Evaluate
    _, y_proba = train_and_evaluate_model(
        X_train_scaled, y_train, X_test_scaled, y_test, RANGE_LABELS
    )

    # 7. Visualize Results
    plot_results(y_test, y_proba, RANGE_LABELS)

if __name__ == "__main__":
    main()
