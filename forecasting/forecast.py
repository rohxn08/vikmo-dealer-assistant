import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Resolve path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALES_PATH = os.path.join(BASE_DIR, "sales_history.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "forecasting", "results.md")

def load_data():
    if not os.path.exists(SALES_PATH):
        raise FileNotFoundError(f"Sales history file not found at {SALES_PATH}")
    df = pd.read_csv(SALES_PATH, parse_dates=["date"])
    return df.sort_values(["sku", "date"]).reset_index(drop=True)

def build_features(df):
    df = df.copy()
    # Seasonal feature
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    
    # Lag features per SKU
    df["lag_1"] = df.groupby("sku")["units_sold"].shift(1)
    df["lag_4"] = df.groupby("sku")["units_sold"].shift(4)
    
    # Rolling feature per SKU
    df["rolling_4"] = df.groupby("sku")["units_sold"].transform(
        lambda x: x.shift(1).rolling(4).mean()
    )
    
    # Drop rows with NaN values resulting from shifts/rolling averages
    return df.dropna().reset_index(drop=True)

def run_naive_baseline(train, test):
    preds = []
    for sku in test["sku"].unique():
        sku_train = train[train["sku"] == sku].sort_values("date")
        # Mean of last 4 weeks in training data for this SKU
        avg = sku_train["units_sold"].tail(4).mean()
        
        sku_test = test[test["sku"] == sku]
        for _, row in sku_test.iterrows():
            preds.append({
                "sku": sku,
                "date": row["date"],
                "actual": row["units_sold"],
                "pred": avg
            })
    return pd.DataFrame(preds)

def train_and_predict():
    print("Loading sales history...")
    df = load_data()
    
    # Define temporal cutoff for last 4 weeks (test set)
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(weeks=4)
    print(f"Dataset range: {df['date'].min().strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    print(f"Temporal cutoff date: {cutoff.strftime('%Y-%m-%d')}")
    
    # Baseline train/test split on raw data (to compute tail 4-week average)
    raw_train = df[df["date"] <= cutoff]
    raw_test = df[df["date"] > cutoff]
    
    # Calculate baseline predictions
    baseline_df = run_naive_baseline(raw_train, raw_test)
    
    # Build features on full dataset to prevent boundary gaps in test lag computation,
    # then split into feature train/test temporally
    df_features = build_features(df)
    
    train_feat = df_features[df_features["date"] <= cutoff]
    test_feat = df_features[df_features["date"] > cutoff]
    
    features = ["week_of_year", "lag_1", "lag_4", "rolling_4", "promo_flag"]
    X_train = train_feat[features]
    y_train = train_feat["units_sold"]
    X_test = test_feat[features]
    y_test = test_feat["units_sold"]
    
    print(f"Training set rows: {len(train_feat)}")
    print(f"Test set rows: {len(test_feat)}")
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on test set
    preds = model.predict(X_test)
    # Clip predictions to 0 since units sold cannot be negative
    preds = np.clip(preds, 0, None)
    
    test_preds = test_feat[["sku", "date", "units_sold"]].copy()
    test_preds["actual"] = test_preds["units_sold"]
    test_preds["pred"] = preds
    
    # Calculate MAE
    baseline_mae = mean_absolute_error(baseline_df["actual"], baseline_df["pred"])
    model_mae = mean_absolute_error(test_preds["actual"], test_preds["pred"])
    improvement = (baseline_mae - model_mae) / baseline_mae * 100
    
    print("\n" + "="*40)
    print(f"Baseline MAE: {baseline_mae:.4f}")
    print(f"Model MAE:    {model_mae:.4f}")
    print(f"Improvement:  {improvement:.2f}%")
    print("="*40 + "\n")
    
    # Calculate per-SKU breakdown
    sku_stats = []
    for sku in df["sku"].unique():
        sku_base = baseline_df[baseline_df["sku"] == sku]
        sku_model = test_preds[test_preds["sku"] == sku]
        
        if len(sku_base) > 0 and len(sku_model) > 0:
            b_mae = mean_absolute_error(sku_base["actual"], sku_base["pred"])
            m_mae = mean_absolute_error(sku_model["actual"], sku_model["pred"])
            imp = ((b_mae - m_mae) / b_mae * 100) if b_mae > 0 else 0
            sku_stats.append({
                "sku": sku,
                "baseline_mae": b_mae,
                "model_mae": m_mae,
                "improvement": imp
            })
            
    write_results_markdown(baseline_mae, model_mae, improvement, sku_stats)
    print(f"Forecasting results written to {RESULTS_PATH}")

def write_results_markdown(baseline_mae, model_mae, improvement, sku_stats):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    
    markdown_content = []
    markdown_content.append("# Demand Forecasting Results\n")
    markdown_content.append("## Overall Performance Comparison\n")
    markdown_content.append("| Model | Mean Absolute Error (MAE) | Improvement |")
    markdown_content.append("|---|---|---|")
    markdown_content.append(f"| Naive Baseline (4-week MA) | {baseline_mae:.4f} | - |")
    markdown_content.append(f"| Linear Regression (with features) | {model_mae:.4f} | **{improvement:.2f}%** |")
    
    markdown_content.append("\n## Per-SKU Breakdown\n")
    markdown_content.append("| SKU | Baseline MAE | Model MAE | Improvement % |")
    markdown_content.append("|---|---|---|---|")
    for s in sku_stats:
        markdown_content.append(f"| {s['sku']} | {s['baseline_mae']:.4f} | {s['model_mae']:.4f} | {s['improvement']:.2f}% |")
        
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_content))

if __name__ == "__main__":
    train_and_predict()
