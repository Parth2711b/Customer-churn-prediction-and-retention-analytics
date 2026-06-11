"""
Customer Churn Risk Score (0-100)  —  v2 (now includes Balance)
================================================================
Input : uploads/churn_cleaned.csv
Output: churn_with_risk_score.csv  (original data + RiskScore + RiskTier)

The score combines 6 fields. Each field contributes points proportional
to how strongly its categories drive churn (from our EDA), plus bonus
points for the two killer interactions we found.

POINT BUDGET (totals exactly 100):
----------------------------------------------------------------------
 Field            Max pts   Why this weight
----------------------------------------------------------------------
 NumOfProducts       28     Strongest single signal (7.6% -> 100% churn)
 Age                 23     2nd strongest (7.6% -> 56% across bands)
 Geography           14     Germany churns at 2x France/Spain
 IsActiveMember      11     26.9% inactive vs 14.3% active
 Balance              8     zero -> 13.8% vs positive -> ~24% churn
 Gender               7     25.1% female vs 16.5% male
 Interactions         9     Inactive senior (+6), Female in Germany (+3)
----------------------------------------------------------------------
 TOTAL              100
"""

import pandas as pd
import numpy as np

df = pd.read_csv("uploads/churn_cleaned.csv")

# ---------------------------------------------------------------
# 1. NumOfProducts  (max 28)   churn: 1->27.7% | 2->7.6% | 3->82.7% | 4->100%
# ---------------------------------------------------------------
prod_pts = df["NumOfProducts"].map({1: 11, 2: 0, 3: 26, 4: 28})

# ---------------------------------------------------------------
# 2. Age  (max 23)   churn: 18-29->7.6% | 30-39->10.9% | 40-49->30.8%
#                            50-59->56.0% | 60+->27.9%
# ---------------------------------------------------------------
age_band = pd.cut(df["Age"], bins=[0, 30, 40, 50, 60, 200],
                  labels=["18_29", "30_39", "40_49", "50_59", "60plus"],
                  right=False)
age_pts = age_band.map({"18_29": 0, "30_39": 3, "40_49": 12,
                        "50_59": 23, "60plus": 10}).astype(int)

# ---------------------------------------------------------------
# 3. Geography  (max 14)   churn: France->16.2% | Spain->16.7% | Germany->32.4%
# ---------------------------------------------------------------
geo_pts = df["Geography"].map({"France": 0, "Spain": 1, "Germany": 14})

# ---------------------------------------------------------------
# 4. IsActiveMember  (max 11)   churn: active->14.3% | inactive->26.9%
# ---------------------------------------------------------------
active_pts = (1 - df["IsActiveMember"]) * 11

# ---------------------------------------------------------------
# 5. Balance  (max 8)   churn: zero->13.8% | 50-100k->19.9%
#                              100-150k->25.8% | 150k+->23.1%
#    Zero balance is protective; risk peaks in the 100k+ range.
# ---------------------------------------------------------------
bal_band = pd.cut(df["Balance"], bins=[-1, 0, 100000, 10**9],
                  labels=["zero", "low", "high"])
balance_pts = bal_band.map({"zero": 0, "low": 5, "high": 8}).astype(int)

# ---------------------------------------------------------------
# 6. Gender  (max 7)   churn: 0 (female)->25.1% | 1 (male)->16.5%
# ---------------------------------------------------------------
gender_pts = (df["Gender"] == 0).astype(int) * 7

# ---------------------------------------------------------------
# 7. Interaction bonuses  (max 9)
#    Inactive & age 50+  -> 81-86% churn observed   (+6)
#    Female in Germany   -> 37.6% churn observed    (+3)
# ---------------------------------------------------------------
inactive_senior_pts = ((df["IsActiveMember"] == 0) & (df["Age"] >= 50)).astype(int) * 6
female_germany_pts  = ((df["Gender"] == 0) & (df["Geography"] == "Germany")).astype(int) * 3

# ---------------------------------------------------------------
# FINAL SCORE + TIERS
# ---------------------------------------------------------------
df["RiskScore"] = (prod_pts + age_pts + geo_pts + active_pts + balance_pts +
                   gender_pts + inactive_senior_pts + female_germany_pts).astype(int)

df["RiskTier"] = pd.cut(df["RiskScore"], bins=[-1, 20, 40, 60, 100],
                        labels=["Low", "Medium", "High", "Critical"])

df.to_csv("churn_with_risk_score.csv", index=False)
print(f"Saved churn_with_risk_score.csv  ({df.shape[0]} rows x {df.shape[1]} cols)")

# ================= VALIDATION =================
print("\nScore range:", df["RiskScore"].min(), "to", df["RiskScore"].max())
print("\nMean RiskScore by actual outcome:")
print(df.groupby("Exited")["RiskScore"].mean().round(1).to_string())

print("\nActual churn rate by RiskTier:")
print(df.groupby("RiskTier", observed=True)["Exited"]
        .agg(churn_rate="mean", customers="count").round(3).to_string())

from sklearn.metrics import roc_auc_score
print(f"\nAUC of RiskScore vs Exited: {roc_auc_score(df['Exited'], df['RiskScore']):.3f}")

# Example profiles
hi = df.loc[df["RiskScore"].idxmax()]
lo = df.loc[df["RiskScore"].idxmin()]
print(f"\nHighest scored customer ({hi['RiskScore']:.0f} pts): "
      f"{hi['Geography']}, Gender={hi['Gender']:.0f}, Age={hi['Age']:.0f}, "
      f"Active={hi['IsActiveMember']:.0f}, Products={hi['NumOfProducts']:.0f}, "
      f"Balance={hi['Balance']:.0f}, Exited={hi['Exited']:.0f}")
print(f"Lowest scored customer ({lo['RiskScore']:.0f} pts): "
      f"{lo['Geography']}, Gender={lo['Gender']:.0f}, Age={lo['Age']:.0f}, "
      f"Active={lo['IsActiveMember']:.0f}, Products={lo['NumOfProducts']:.0f}, "
      f"Balance={lo['Balance']:.0f}, Exited={lo['Exited']:.0f}")
