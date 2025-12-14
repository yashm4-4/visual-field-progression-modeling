# Random Forest Regression - Glaucoma Progression 

# CELL 1

# Importing libraries and loading the dataset (UW)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,

)

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (7, 5)
plt.rcParams["font.size"] = 12

# Loading UW visual field dataset:

uw_vf = pd.read_csv("UW_VF_Data.csv")
print("UW_VF_Data shape:", uw_vf.shape)

# pattern deviation (PD) columns:

pd_cols = [c for c in uw_vf.columns if c.startswith("PD_")]
print("Number of PD locations:", len(pd_cols))

# converting PD values to numeric:

for c in pd_cols:
    uw_vf[c] = pd.to_numeric(uw_vf[c], errors="coerce")

# num -1 == “no response” ==> setting to NaN:

uw_vf["has_no_response"] = uw_vf[pd_cols].eq(-1).any(axis=1).astype(int)
uw_vf[pd_cols] = uw_vf[pd_cols].replace(-1, np.nan)

# recomputing mean sensitivity (MS) from PD:

uw_vf["MS_recomputed"] = uw_vf[pd_cols].mean(axis=1, skipna=True)

# simple PD summaries per visit:

uw_vf["PD_std"] = uw_vf[pd_cols].std(axis=1, skipna=True)
uw_vf["PD_min"] = uw_vf[pd_cols].min(axis=1, skipna=True)
uw_vf["PD_max"] = uw_vf[pd_cols].max(axis=1, skipna=True)
uw_vf["PD_median"] = uw_vf[pd_cols].median(axis=1, skipna=True)
uw_vf["missing_PD_count"] = uw_vf[pd_cols].isna().sum(axis=1)

# eye ID and time axis:

uw_vf["Eye_ID"] = uw_vf["PatID"].astype(str) + "_" + uw_vf["Eye"].astype(str)
uw_vf["time_years"] = pd.to_numeric(uw_vf["Time_from_Baseline"], errors="coerce")

# CELL 2

# Computing MS slope (progression rate) per eye:

def summarize_eye(df_eye):
    
    """Fit MS_recomputed vs time_years for one eye."""


    df_eye = df_eye.dropna(subset=["time_years", "MS_recomputed"])
    n = df_eye.shape[0]
    if n < 3:
        return pd.Series({
            "MS_slope": np.nan,
            "n_visits": n,
            "fit_r2": np.nan,
        })
    
    X = df_eye[["time_years"]].values
    y = df_eye["MS_recomputed"].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    
    return pd.Series({
        "MS_slope": model.coef_[0],
        "n_visits": n,
        "fit_r2": r2,
    })

# summarizing per eye:

slopes = uw_vf.groupby("Eye_ID").apply(summarize_eye).reset_index()
print("Total eyes:", slopes.shape[0])

# keepig eyes with decent fits and enough visits:

mask_good = (slopes["n_visits"] >= 5) & (slopes["fit_r2"] >= 0.2)
slopes_good = slopes[mask_good].copy()

print("Good-quality eyes:", slopes_good.shape[0])
print("Mean slope (good eyes):", slopes_good["MS_slope"].mean().round(3))

# merging back to main table:

uw_vf = uw_vf.merge(
    slopes_good[["Eye_ID", "MS_slope", "n_visits", "fit_r2"]],
    on="Eye_ID",
    how="inner"
)
print("Rows after merging:", uw_vf.shape[0])

# CELL 3 


# Picking baseline visit for each eye:

idx_baseline = uw_vf.groupby("Eye_ID")["time_years"].idxmin()
uw_base = uw_vf.loc[idx_baseline].copy()

# Baseline MS and encodings

uw_base["baseline_MS"] = uw_base["MS_recomputed"]

uw_base["Gender_code"] = uw_base["Gender"].map({"F": 0, "M": 1})
uw_base["Eye_code"] = uw_base["Eye"].map({"Left": 0, "Right": 1})

# PD columns

pd_cols = [c for c in uw_base.columns if c.startswith("PD_")]

# Core columns:

base_cols_wanted = [
    "Eye_ID",
    "MS_slope",
    "n_visits",
    "fit_r2",
    "baseline_MS",   
    "Age",           
    "Gender_code",
    "Eye_code",
    "PD_std",
    "PD_min",
    "PD_max",
    "PD_median",
    "missing_PD_count",
    "has_no_response",
]

base_cols_available = [c for c in base_cols_wanted if c in uw_base.columns]
print("Base cols kept:", base_cols_available)

# Building reg_df = baseline clinical + PD locations:

reg_df = uw_base[base_cols_available].copy()
reg_df = pd.concat([reg_df, uw_base[pd_cols]], axis=1)

# 6) Drop rows with missing slope:

reg_df = reg_df.dropna(subset=["MS_slope"])
print("reg_df shape:", reg_df.shape)

# Defining feature_cols = everything except ID and target:

feature_cols = [c for c in reg_df.columns if c not in ["Eye_ID", "MS_slope"]]
print("Number of features:", len(feature_cols))

# CELL 4: 

mask_2plus = reg_df["n_visits"] >= 7
reg_df_2 = reg_df[mask_2plus].copy()

print(f"Eyes with ≥ 7 visits: {reg_df_2.shape[0]} (out of {reg_df.shape[0]})")

available_features = [f for f in feature_cols if f in reg_df_2.columns]
missing_features = [f for f in feature_cols if f not in reg_df_2.columns]

print("Using", len(available_features), "features.")
if missing_features:
    print("Skipping missing features:", missing_features)

X_reg = reg_df_2[available_features].copy()  
y_reg = reg_df_2["MS_slope"].values           
n_visits_reg = reg_df_2["n_visits"].values    

# Droping columns that are entirely NaN:

all_nan_cols = X_reg.columns[X_reg.isna().all()]
if len(all_nan_cols) > 0:
    print("Dropping all-NaN columns:", list(all_nan_cols))
    X_reg_clean = X_reg.drop(columns=all_nan_cols)
else:
    X_reg_clean = X_reg.copy()

feature_cols_clean = X_reg_clean.columns
print("Final feature count after cleanup:", len(feature_cols_clean))

# Median imputation:

imputer = SimpleImputer(strategy="median")
X_imp_array = imputer.fit_transform(X_reg_clean)
X_imp = pd.DataFrame(X_imp_array, columns=feature_cols_clean, index=X_reg_clean.index)

# 3) Train/test split: 80/20 %:

X_train, X_test, y_train, y_test, nvis_train, nvis_test = train_test_split(
    X_imp,
    y_reg,
    n_visits_reg,
    test_size=0.2,
    random_state=42,
)

# Random Forest Regressor Model:

rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=15,
    min_samples_leaf=3,
    min_samples_split=5,
    max_features=0.5,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Metrics:

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
corr = np.corrcoef(y_test, y_pred)[0, 1]

print("Random Forest performance (≥ 7 visits):")
print(f"  MAE  : {mae:.3f} dB/year")
print(f"  RMSE : {rmse:.3f} dB/year")
print(f"  R²   : {r2:.3f}")
print(f"  Corr : {corr:.3f}")


# CELL 5:  Visualizations

# Features

importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols_clean).sort_values(ascending=False)

plt.figure(figsize=(7, 5))
feat_imp.head(15).plot(kind="barh")

plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Physiological Features Predicting Glaucoma Progression (≥ 7 visits)")
plt.tight_layout()
plt.show()


# classification based on slopes to Stable / Slow / Fast:

def slope_to_class(s):
    if s > -0.25:
        return "Stable"
    elif s > -0.55:
        return "Slow"
    else:
        return "Fast"

true_classes = np.array([slope_to_class(s) for s in y_test])
pred_classes = np.array([slope_to_class(s) for s in y_pred])

print("\nClassification report (≥ 7 visits):")
print(classification_report(true_classes, pred_classes, labels=["Stable", "Slow", "Fast"]))

cm = confusion_matrix(true_classes, pred_classes, labels=["Stable", "Slow", "Fast"])

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d",
    cmap="Blues",
    xticklabels=["Stable", "Slow", "Fast"],
    yticklabels=["Stable", "Slow", "Fast"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (≥ 7 visits)")
plt.tight_layout()
plt.show()



# Predicted vs True scatter plot

plt.figure(figsize=(6, 6))
plt.scatter(
    y_test,
    y_pred,
    c=nvis_test,
    cmap="viridis",
    s=30,
    alpha=0.7
)

# perfect prediction line:

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction Line")


# plot labels:

plt.xlabel("Actual MS slope (dB/year)")
plt.ylabel("Predicted MS slope (dB/year)")
plt.title("Predicted vs Actual MS Slopes (≥ 7 visits)")


plt.colorbar(label="Number of visits")
plt.grid(alpha=0.5)
plt.legend()
plt.tight_layout()

plt.show()



# Violin Plot

df = slopes[["MS_slope", "n_visits"]].dropna()

bins = [3, 5, 7, 9, 100]

labels = ["3 – 4 Visits", "5 – 6 Visits", "7 – 8 Visits", "9+ Visits"]

df["visit_bin"] = pd.cut(df["n_visits"], bins=bins, labels=labels, right=False)

palette_colors = ["#0C64F1", "#0ABF53", "#F25C5C", "#A68CFF"]

plt.figure(figsize=(6, 5))
sns.violinplot(
    data=df,
    x="visit_bin",
    y="MS_slope",
    palette=palette_colors,
    inner="quartile",
    cut=0,
)

plt.ylim(-5, 5)
plt.axhline(0, ls="--", color="black", alpha=0.5)

plt.xlabel("Number of visits per eye")
plt.ylabel("MS slope (dB/year)")

plt.title("MS slope vs. follow-up visits")
plt.tight_layout()
plt.show()




