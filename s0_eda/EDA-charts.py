import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from s1_data.a0_setup_directories import *

df = pd.read_csv("data/train_after_imputation_EDA.csv")


# 1. preliminary interaction terms creation

# Share of above-ground living area relative to total area
df["FinishedAreaPct"] = df["GrLivArea"] / (
    df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
)
# Living space × number of rooms
df["Living_Rooms"] = df["GrLivArea"] * df["TotRmsAbvGrd"]

# Garage capacity proxy (area × number of cars)
df["Garage_Space"] = df["GarageArea"] * df["GarageCars"]

# Garage age in years (safe handling of missing/no-garage cases)
df["Age_Garage"] = np.where(
    (df["GarageYrBlt"].isna()) |
    (df["GarageType"] == "no_garage") |
    ((df["YrSold"] - df["GarageYrBlt"]) < 0),
    0,
    df["YrSold"] - df["GarageYrBlt"]
)

df["Garage_AgeCars"] = df["Age_Garage"] * df["GarageCars"]

# House age in years (safe handling of invalid values)
df["Age_House"] = np.where(
    (df["YrSold"].isna()) |
    (df["YearBuilt"].isna()) |
    ((df["YrSold"] - df["YearBuilt"]) < 0),
    0,
    df["YrSold"] - df["YearBuilt"]
)

# Porch intensity adjusted by house age (non-linear scaling)
df["Porch_Age"] = np.cbrt(df["EnclosedPorch"]) * df["Age_House"]

# Room composition ratios
df["Ratio_Bedroom_Rooms"] = df["BedroomAbvGr"] / df["TotRmsAbvGrd"]

# Upper floor contribution relative to total living area
df["Ratio_2ndFlr_Living"] = df["2ndFlrSF"] / df["GrLivArea"]

# Time since last remodel
df["Yrs_Since_Remodel"] = np.where(
    (df["YrSold"].isna()) |
    (df["YearRemodAdd"].isna()) |
    ((df["YrSold"] - df["YearRemodAdd"]) < 0),
    0,
    df["YrSold"] - df["YearRemodAdd"]
)


con_numeric = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', "Age_House", "Yrs_Since_Remodel",
        "FinishedAreaPct", "Living_Rooms", "Garage_Space", "Age_Garage", "Garage_AgeCars",
        "Porch_Age", "Ratio_Bedroom_Rooms", "Ratio_2ndFlr_Living"
       ]

target = "SalePrice" 


# Continuous numeric vs SalePrice
for col in con_numeric:
    plt.figure(figsize=(12,5))
    
    # Left: raw
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x=col, y=target)
    plt.xlabel(col)
    plt.ylabel(target)
    plt.title(f'{col} vs {target} (Raw)')
    
    # Right: log-transformed
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x=np.log1p(df[col]), y=target)
    plt.xlabel(f'log({col}+1)')
    plt.ylabel(target)
    plt.title(f'{col} vs {target} (Log Transformed)')
    
    plt.tight_layout()
    # save file (safe filename)
    save_path = os.path.join(plot_dir, f"01_{col}_vs_{target}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# 2. finalized variables to transform and create interactions
df = pd.read_csv("data/train_after_imputation_EDA.csv")
# -------------------------
# Helper features
# -------------------------

# Age features
df["Age_House"] = np.where(
    (df["YrSold"].isna()) |
    (df["YearBuilt"].isna()) |
    ((df["YrSold"] - df["YearBuilt"]) < 0),
    0,
    df["YrSold"] - df["YearBuilt"]
)

df["Age_Garage"] = np.where(
    (df["GarageYrBlt"].isna()) |
    (df["GarageType"] == "no_garage") |
    ((df["YrSold"] - df["GarageYrBlt"]) < 0),
    0,
    df["YrSold"] - df["GarageYrBlt"]
)

df["Yrs_Since_Remodel"] = np.where(
    (df["YrSold"].isna()) |
    (df["YearRemodAdd"].isna()) |
    ((df["YrSold"] - df["YearRemodAdd"]) < 0),
    0,
    df["YrSold"] - df["YearRemodAdd"]
)

# -------------------------
# Log transformations
# -------------------------
df["log_LotFrontage"] = np.log1p(df["LotFrontage"])
df["log_LotArea"] = np.log1p(df["LotArea"])
df["log_1stFlrSF"] = np.log1p(df["1stFlrSF"])
df["log_2ndFlrSF"] = np.log1p(df["2ndFlrSF"])
df["log_LowQualFinSF"] = np.log1p(df["LowQualFinSF"])
df["log_GrLivArea"] = np.log1p(df["GrLivArea"])
df["log_Yrs_Since_Remodel"] = np.log1p(df["Yrs_Since_Remodel"])
df["log_Age_Garage"] = np.log1p(df["Age_Garage"])

# -------------------------
# Square root transformations
# -------------------------
df["sqrt_TotalBsmtSF"] = np.sqrt(df["TotalBsmtSF"])
df["sqrt_WoodDeckSF"] = np.sqrt(df["WoodDeckSF"])
df["sqrt_BsmtUnfSF"] = np.sqrt(df["BsmtUnfSF"])
df["sqrt_BsmtFinSF1"] = np.sqrt(df["BsmtFinSF1"])

# -------------------------
# Cube root transformations
# -------------------------
df["cbrt_MasVnrArea"] = np.cbrt(df["MasVnrArea"])
df["cbrt_OpenPorchSF"] = np.cbrt(df["OpenPorchSF"])

# -------------------------
# Interaction terms
# -------------------------

# Finished area percentage
df["FinishedAreaPct"] = df["GrLivArea"] / (
    df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
)

# log(1 + GrLivArea * TotRmsAbvGrd)
df["Living_Rooms"] = np.log1p(df["GrLivArea"] * df["TotRmsAbvGrd"])

# log(1 + GarageArea * GarageCars)
df["Garage_Space"] = np.log1p(df["GarageArea"] * df["GarageCars"])

# log(1 + Age_Garage * GarageCars)
df["Garage_AgeCars"] = np.log1p(df["Age_Garage"] * df["GarageCars"])

# log(1 + cbrt(EnclosedPorch) * Age_House)
df["Porch_Age"] = np.log1p(np.cbrt(df["EnclosedPorch"]) * df["Age_House"])

# ratios (same as SQL)
df["Ratio_Bedroom_Rooms"] = df["BedroomAbvGr"] / df["TotRmsAbvGrd"]
df["Ratio_2ndFlr_Living"] = df["2ndFlrSF"] / df["GrLivArea"]



# 3. correlation matrix
con_numeric_transformed = [
    'BsmtFinSF2',
    'GarageArea',
    'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
    'PoolArea',
    'MiscVal',

    # -------------------------
    # Engineered time features
    # -------------------------
    'Age_House',
    'Age_Garage',
    'Yrs_Since_Remodel',

    # -------------------------
    # Transformed (log / sqrt / cbrt versions)
    # -------------------------
    'log_LotFrontage',
    'log_LotArea',
    'log_1stFlrSF',
    'log_2ndFlrSF',
    'log_LowQualFinSF',
    'log_GrLivArea',
    'log_Yrs_Since_Remodel',
    'log_Age_Garage',

    'sqrt_TotalBsmtSF',
    'sqrt_WoodDeckSF',
    'sqrt_BsmtUnfSF',
    'sqrt_BsmtFinSF1',

    'cbrt_MasVnrArea',
    'cbrt_OpenPorchSF',

    # -------------------------
    # Interaction features
    # -------------------------
    'FinishedAreaPct',
    'Living_Rooms',
    'Garage_Space',
    'Garage_AgeCars',
    'Porch_Age',
    'Ratio_Bedroom_Rooms',
    'Ratio_2ndFlr_Living'
]
discrete_numeric = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageCars']



corr_features = con_numeric_transformed + discrete_numeric + [target]
corr_df = df[corr_features]
corr_matrix = corr_df.corr()
plt.figure(figsize=(20, 16))  # Increase the figure size
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5,
            annot_kws={"size": 10})  # Decrease the annotation font size
plt.title("Correlation Matrix")
plt.savefig(os.path.join(plot_dir, "02_correlation_matrix.png"))  # Save the plot to a file
plt.close()



# 3. Bar chart for categorical columns
categorical_cols = ['MSSubClass','MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 
       'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition']

for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, data=df)
    plt.title(f"Bar Chart of {col}")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(plot_dir, f"03_bar_plot_{col}.png"))
    plt.close()


# 4. Histogram + Density chart and QQ plot for numerical columns
for col in con_numeric_transformed + discrete_numeric:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[col], kde=True)
    plt.title(f"Histogram and Density Plot of {col}")
    plt.savefig(os.path.join(plot_dir, f"04_histogram_density_plot_{col}.png"))
    plt.close()

for col in con_numeric_transformed + discrete_numeric:
    plt.figure(figsize=(10, 5))
    stats.probplot(df[col], dist="norm", plot=plt)
    plt.title(f"QQ Plot of {col}")
    plt.savefig(os.path.join(plot_dir, f"05_QQ_plot_{col}.png"))
    plt.close()


# 5. Outlier detection: the partial sales dropped in s1_data/a2_drop_outliers
"""
Reads train.csv directly instead of train_after_imputation_EDA.csv, because the
latter is exported by a4 after a2 has already dropped these rows.

Influence diagnostics from an OLS of log(SalePrice) on log(GrLivArea) and
OverallQual. The three metrics answer different questions, and a row is only
worth dropping when they agree.

leverage (hat matrix diagonal)
    How unusual a row's predictors are, ignoring SalePrice entirely. It is the
    weight a row carries in its own fitted value, averaging p/n = 0.002 here.
    High leverage means a row *could* move the fit, not that it does: Id 534
    carries the highest leverage in the data at 0.0149, purely for being the
    smallest house at 334 sqft against a median of 1464, yet it sits on the
    trend and shifts nothing.

stud_resid (externally studentized residual)
    How far SalePrice lands from its prediction, scaled by an error variance
    re-estimated with that row held out. Holding it out is the point: a row that
    drags the fit toward itself would otherwise shrink its own residual and go
    unnoticed. Roughly t-distributed, so anything past +/-3 is a real anomaly.

cooks_d (Cook's distance)
    How far every fitted value moves when the row is deleted, combining the two
    above. Leverage says a row could matter and the residual says it disagrees
    with the model, but only Cook's D says removing it would actually change the
    coefficients.

All four homes over 4000 sqft carry roughly 5x the average leverage on size
alone, which is why leverage cannot single out the bad ones. Only 524 and 1299
pair that with extreme residuals (-5.5 and -6.7), so only they reach a high
Cook's D, ranking 1st and 2nd out of 1460 rows against 12th and 16th for the two
that sold at market.
"""
df_raw = pd.read_csv("data/train.csv")

X_raw = sm.add_constant(pd.DataFrame({
    "log_GrLivArea": np.log(df_raw["GrLivArea"]),
    "OverallQual": df_raw["OverallQual"],
}))
influence = sm.OLS(np.log(df_raw["SalePrice"]), X_raw).fit().get_influence()
df_raw["leverage"] = influence.hat_matrix_diag
df_raw["stud_resid"] = influence.resid_studentized_external
df_raw["cooks_d"] = influence.cooks_distance[0]

large = df_raw["GrLivArea"] > 4000
dropped = large & (df_raw["SalePrice"] < 300000)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: both underpriced mansions are Partial sales, the two at market are not
sns.scatterplot(data=df_raw, x="GrLivArea", y="SalePrice", hue="SaleCondition",
                alpha=0.6, ax=axes[0])
sns.regplot(data=df_raw, x="GrLivArea", y="SalePrice", scatter=False, lowess=True,
            color="black", line_kws={"linewidth": 1}, ax=axes[0])
axes[0].axvline(4000, color="grey", linestyle="--", linewidth=1)
# Ids 692 and 1183 nearly coincide, so stagger the labels vertically
offsets = [12 if i % 2 == 0 else -20 for i in range(large.sum())]
for (_, row), dy in zip(df_raw[large].iterrows(), offsets):
    axes[0].annotate(f"Id {int(row['Id'])} ({row['SaleCondition']})",
                     (row["GrLivArea"], row["SalePrice"]),
                     textcoords="offset points", xytext=(-30, dy), fontsize=9)
axes[0].set_title("GrLivArea vs SalePrice, homes over 4000 sqft labelled")

# Right: the two dropped rows are the most influential points in the training set
sns.scatterplot(data=df_raw, x="leverage", y="stud_resid", size="cooks_d",
                sizes=(10, 400), alpha=0.4, legend=False, ax=axes[1])
axes[1].axhline(0, color="grey", linewidth=1)
for bound in (-3, 3):
    axes[1].axhline(bound, color="red", linestyle="--", linewidth=1)
for (_, row), dy in zip(df_raw[large].iterrows(), offsets):
    axes[1].annotate(f"Id {int(row['Id'])}", (row["leverage"], row["stud_resid"]),
                     textcoords="offset points", xytext=(10, dy / 2), fontsize=9)
axes[1].set_title("Studentized residual vs leverage, sized by Cook's D")

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "06_outlier_detection.png"), dpi=300, bbox_inches="tight")
plt.close()

cooks_rank = df_raw["cooks_d"].rank(ascending=False).astype(int)
summary = df_raw.loc[large, ["Id", "GrLivArea", "SalePrice", "SaleCondition", "stud_resid", "cooks_d"]]
summary["cooks_rank"] = cooks_rank[large]
summary["dropped"] = dropped[large]
print("Homes over 4000 sqft, ranked by influence out of "
      f"{len(df_raw)} training rows:")
print(summary.to_string(index=False))

