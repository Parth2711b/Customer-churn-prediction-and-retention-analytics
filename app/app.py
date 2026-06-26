import matplotlib
matplotlib.use('Agg')
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

model = joblib.load('churn_model.pkl')
X_test_transformed_saved = np.load('X_test_transformed.npy')
y_test_saved = pd.read_csv('y_test.csv').squeeze()

st.set_page_config(
    page_title="FintelliQ — Churn Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Geist', 'Inter', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.stApp { background-color: #0a0a0a !important; }
.block-container { padding: 32px 32px !important; max-width: 1200px !important; }

section[data-testid="stSidebar"] {
    background-color: #131315 !important;
    border-right: 1px solid #262629 !important;
    min-width: 300px !important;
    max-width: 300px !important;
    transform: none !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p { color: #8a8a93 !important; font-size: 13px !important; }

.stSelectbox > div > div {
    background-color: #0a0a0a !important;
    border: 1px solid #262629 !important;
    color: #f4f4f5 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}
.stNumberInput > div > div > input {
    background-color: #0a0a0a !important;
    border: 1px solid #262629 !important;
    color: #f4f4f5 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #f4f4f5 !important;
    border: 2px solid #0a0a0a !important;
}
div[data-testid="stMetric"] {
    background-color: #0a0a0a !important;
    border: 1px solid #262629 !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    text-align: center !important;
}
div[data-testid="stMetricLabel"] { color: #8a8a93 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.05em; }
div[data-testid="stMetricValue"] { color: #f4f4f5 !important; font-size: 32px !important; font-weight: 700 !important; }
div[data-testid="stAlert"] { background-color: #131315 !important; border: 1px solid #262629 !important; border-radius: 8px !important; }
details { background-color: #131315 !important; border: 1px solid #262629 !important; border-radius: 8px !important; }
details summary { color: #8a8a93 !important; font-size: 13px !important; }
th { background-color: #131315 !important; color: #8a8a93 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.05em; padding: 10px 14px !important; border-bottom: 1px solid #262629 !important; }
td { color: #8a8a93 !important; font-size: 13px !important; padding: 10px 14px !important; border-bottom: 1px solid #131315 !important; }
hr { border-color: #262629 !important; margin: 24px 0 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #262629; border-radius: 3px; }

section[data-testid="stSidebar"][aria-expanded="false"] {
    display: block !important;
    width: 300px !important;
}
div[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.markdown("""
<div style="display:flex; align-items:center; gap:12px; padding-bottom:20px; border-bottom:1px solid #262629; margin-bottom:8px;">
    <div style="width:36px; height:36px; border-radius:8px; border:1px solid #262629; background:#1c1c1f;
                display:flex; align-items:center; justify-content:center; font-size:16px; font-weight:800; color:#f4f4f5;">F</div>
    <div>
        <div style="font-size:15px; font-weight:800; color:#f4f4f5; letter-spacing:-0.02em;">FintelliQ</div>
        <div style="font-size:11px; color:#8a8a93; font-weight:500;">Customer Churn Intelligence</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<p style='font-size:11px; font-weight:600; color:#f4f4f5; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;'>Customer Profile</p>", unsafe_allow_html=True)
age          = st.sidebar.slider("Age", 18, 92, 38)
credit_score = st.sidebar.slider("Credit Score", 500, 850, 680)
tenure       = st.sidebar.slider("Tenure (years)", 0, 10, 3)
balance      = st.sidebar.number_input("Account Balance (₹)", 0, 500000, 50000, step=1000)

st.sidebar.markdown("<p style='font-size:11px; font-weight:600; color:#f4f4f5; text-transform:uppercase; letter-spacing:0.06em; margin-top:8px; margin-bottom:4px;'>Engagement</p>", unsafe_allow_html=True)
num_products     = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
estimated_salary = st.sidebar.number_input("Estimated Salary (₹)", 10000, 500000, 100000, step=5000)
is_active        = st.sidebar.selectbox("Active Member", ["Yes", "No"])
gender           = st.sidebar.selectbox("Gender", ["Male", "Female"])
geography        = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])

# ── Feature engineering ───────────────────────────────────
is_active_val        = 1 if is_active == "Yes" else 0
gender_val           = 1 if gender == "Male" else 0
is_zero_balance      = 1 if balance == 0 else 0
balance_salary_ratio = balance / estimated_salary if estimated_salary > 0 else 0
is_germany           = 1 if geography == "Germany" else 0
is_high_risk         = 1 if (num_products >= 3 and is_active_val == 0) else 0
is_inactive_senior   = 1 if (age > 60 and is_active_val == 0) else 0
is_female_germany    = 1 if (gender == "Female" and geography == "Germany") else 0
age_group = "Young" if age < 30 else "Mid" if age < 45 else "Senior" if age < 60 else "Old"

input_df = pd.DataFrame([{
    'CreditScore': credit_score, 'Geography': geography, 'Gender': gender_val,
    'Age': age, 'Tenure': tenure, 'Balance': balance, 'NumOfProducts': num_products,
    'IsActiveMember': is_active_val, 'IsZeroBalance': is_zero_balance, 'AgeGroup': age_group,
    'IsHighRiskProduct': is_high_risk, 'IsGermany': is_germany,
    'BalanceToSalaryRatio': balance_salary_ratio, 'IsInactiveSenior': is_inactive_senior,
    'IsFemaleGermany': is_female_germany
}])

prob       = model.predict_proba(input_df)[0][1]
threshold  = 0.35
prediction = 1 if prob >= threshold else 0
margin     = 0.08
lower      = max(0, prob - margin)
upper      = min(1, prob + margin)

explainer         = shap.TreeExplainer(model.named_steps['model'])
input_transformed = model.named_steps['preprocessor'].transform(input_df)
shap_values       = explainer.shap_values(input_transformed)
feature_names     = model.named_steps['preprocessor'].get_feature_names_out()

name_map = {
    'num__Age': 'Age', 'num__NumOfProducts': 'Number of Products',
    'num__IsActiveMember': 'Active Member', 'num__Gender': 'Gender',
    'num__Balance': 'Account Balance', 'num__CreditScore': 'Credit Score',
    'num__Tenure': 'Tenure', 'num__IsZeroBalance': 'Zero Balance',
    'num__IsInactiveSenior': 'Inactive Senior', 'num__IsHighRiskProduct': 'High Risk Product',
    'num__IsGermany': 'Is Germany', 'num__BalanceToSalaryRatio': 'Balance to Salary Ratio',
    'num__IsFemaleGermany': 'Female + Germany', 'cat__Geography_Germany': 'Geography: Germany',
    'cat__Geography_Spain': 'Geography: Spain', 'cat__AgeGroup_Young': 'Age Group: Young',
    'cat__AgeGroup_Mid': 'Age Group: Mid', 'cat__AgeGroup_Senior': 'Age Group: Senior',
    'cat__AgeGroup_Old': 'Age Group: Old',
}

shap_df = pd.DataFrame({'Feature': feature_names, 'Impact': shap_values[0]})
shap_df['Feature'] = shap_df['Feature'].map(name_map).fillna(shap_df['Feature'])
shap_df = shap_df.sort_values('Impact', key=abs, ascending=False).head(8)
shap_df = shap_df.sort_values('Impact', ascending=True)
top_risk = shap_df[shap_df['Impact'] > 0].sort_values('Impact', ascending=False).head(3)['Feature'].tolist()

if prob < 0.3:
    risk_label, border_color, text_color, ring_color = "Low Risk", "#166534", "#22c55e", "rgba(34,197,94,0.15)"
elif prob < 0.5:
    risk_label, border_color, text_color, ring_color = "Medium Risk", "#713f12", "#eab308", "rgba(234,179,8,0.15)"
elif prob < 0.7:
    risk_label, border_color, text_color, ring_color = "High Risk", "#7c2d12", "#f97316", "rgba(249,115,22,0.15)"
else:
    risk_label, border_color, text_color, ring_color = "Critical Risk", "#7f1d1d", "#ef4444", "rgba(239,68,68,0.15)"

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:24px;">
    <h1 style="font-size:22px; font-weight:700; color:#f4f4f5; letter-spacing:-0.02em; margin:0;">Churn Risk Assessment</h1>
    <p style="font-size:13px; color:#8a8a93; margin-top:4px;">Adjust customer profile in the sidebar to update predictions in real time.</p>
</div>
""", unsafe_allow_html=True)

# ── Section 1: Risk Summary ───────────────────────────────
recommendation_map = {
    'Number of Products':      ("Simplify product portfolio", "Too many products correlates with dissatisfaction. Offer a portfolio review."),
    'Age':                     ("Assign relationship manager", "Older customers respond well to personalized, dedicated service."),
    'Active Member':           ("Re-engagement campaign", "Inactive members convert 2x with targeted cashback or fee waiver offers."),
    'Inactive Senior':         ("Priority senior outreach", "Inactive senior customers are highest risk. Offer assisted banking services."),
    'Geography: Germany':      ("Escalate to regional team", "Germany has the highest churn rate in this dataset."),
    'Account Balance':         ("Offer premium savings product", "High balance + churn risk signals dissatisfaction with current returns."),
    'Balance to Salary Ratio': ("Competitive interest rate offer", "Customer may be seeking better returns elsewhere."),
    'Gender':                  ("Targeted loyalty program", "Female customers show slightly higher churn tendency in this segment."),
    'Female + Germany':        ("Immediate outreach required", "This customer falls into the single highest-risk segment."),
    'High Risk Product':       ("Schedule product review call", "Multiple products + inactivity signals high dissatisfaction risk."),
    'Credit Score':            ("Financial advisory referral", "Lower credit score may indicate financial stress."),
    'Zero Balance':            ("Account reactivation incentive", "Zero balance is a strong leading indicator of imminent churn."),
}

recos = []
for feature in top_risk:
    if feature == 'Zero Balance' and balance > 0:
        continue
    if feature == 'Number of Products' and num_products < 3:
        continue
    if feature in recommendation_map:
        recos.append(recommendation_map[feature])
if not recos:
    recos = [("Monitor quarterly", "No immediate risk factors identified for this customer.")]

recos_html = ""
for i, (title, desc) in enumerate(recos, 1):
    recos_html += f"""
    <li style="display:flex; align-items:flex-start; gap:12px;">
        <span style="flex-shrink:0; width:20px; height:20px; border-radius:6px; border:1px solid #262629;
                     background:#1c1c1f; font-size:11px; font-weight:600; color:#8a8a93;
                     display:flex; align-items:center; justify-content:center;">{i}</span>
        <span>
            <strong style="display:block; font-size:13px; font-weight:500; color:#f4f4f5;">{title}</strong>
            <span style="font-size:12px; color:#8a8a93;">{desc}</span>
        </span>
    </li>
    """

comparable_html = ""
age_col_idx      = list(feature_names).index('num__Age')
products_col_idx = list(feature_names).index('num__NumOfProducts')
mask = (
    (np.abs(X_test_transformed_saved[:, age_col_idx] - age) < 10) &
    (X_test_transformed_saved[:, products_col_idx] == num_products)
)
similar_count = mask.sum()
if similar_count > 10:
    similar_churn_rate = y_test_saved.values[mask].mean()
    comparable_html = f"""
    <p style="margin-top:16px; font-size:12px; color:#8a8a93; padding-top:14px; border-top:1px solid #262629;">
        Among <strong style="color:#f4f4f5;">{similar_count}</strong> customers with similar profile —
        <strong style="color:{text_color};">{similar_churn_rate:.0%}</strong> churned historically.
    </p>
    """

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(f"""
    <div style="background:#131315; border:1px solid #262629; border-left:3px solid {border_color};
                border-radius:12px; padding:22px 24px;">
        <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:12px;">
            <p style="font-size:13px; font-weight:500; color:#8a8a93; margin:0;">Churn Probability</p>
            <span style="display:inline-flex; align-items:center; gap:6px; border-radius:6px; padding:4px 10px;
                         font-size:11px; font-weight:600; color:{text_color}; background:{ring_color};">
                <span style="width:6px; height:6px; border-radius:50%; background:{text_color};"></span>
                {risk_label}
            </span>
        </div>
        <div style="display:flex; align-items:baseline; gap:8px; margin-top:14px;">
            <span style="font-size:52px; font-weight:700; line-height:1; letter-spacing:-0.03em; color:#f4f4f5;">{prob*100:.0f}</span>
            <span style="font-size:22px; font-weight:600; color:#8a8a93;">%</span>
        </div>
        <p style="font-size:12px; color:#8a8a93; margin-top:14px;">
            Confidence interval {lower:.0%} – {upper:.0%} &middot; XGBoost v1.7
        </p>
        {comparable_html}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="background:#131315; border:1px solid #262629; border-radius:12px; padding:22px 24px;">
        <div style="display:flex; align-items:center; justify-content:space-between;">
            <p style="font-size:13px; font-weight:500; color:#8a8a93; margin:0;">Retention Recommendations</p>
            <span style="font-size:12px; color:#8a8a93;">{len(recos)} actions</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    for i, (title, desc) in enumerate(recos, 1):
        st.markdown(f"""
        <div style="display:flex; align-items:flex-start; gap:12px; padding:10px 24px; background:#131315; border-left:1px solid #262629; border-right:1px solid #262629;">
            <span style="flex-shrink:0; width:20px; height:20px; border-radius:6px; border:1px solid #262629;
                         background:#1c1c1f; font-size:11px; font-weight:600; color:#8a8a93;
                         display:flex; align-items:center; justify-content:center;">{i}</span>
            <span>
                <strong style="display:block; font-size:13px; font-weight:500; color:#f4f4f5;">{title}</strong>
                <span style="font-size:12px; color:#8a8a93;">{desc}</span>
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background:#131315; border:1px solid #262629; border-top:none; border-radius:0 0 12px 12px; height:16px;"></div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

# ── Section 2: SHAP Chart ─────────────────────────────────
st.markdown("""
<div style="background:#131315; border:1px solid #262629; border-radius:12px; padding:22px 24px 8px 24px;">
    <div style="display:flex; align-items:center; justify-content:space-between; padding-bottom:16px; border-bottom:1px solid #262629; margin-bottom:4px;">
        <div>
            <h2 style="font-size:13px; font-weight:500; color:#f4f4f5; margin:0;">Why this prediction?</h2>
            <p style="font-size:12px; color:#8a8a93; margin-top:2px;">SHAP feature contributions for this customer</p>
        </div>
        <div style="display:flex; gap:16px;">
            <div style="display:flex; align-items:center; gap:6px; font-size:12px; color:#8a8a93;">
                <span style="width:10px; height:10px; border-radius:3px; background:#ef4444; display:inline-block;"></span>Increases risk
            </div>
            <div style="display:flex; align-items:center; gap:6px; font-size:12px; color:#8a8a93;">
                <span style="width:10px; height:10px; border-radius:3px; background:#22c55e; display:inline-block;"></span>Decreases risk
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

colors = ['#ef4444' if x > 0 else '#22c55e' for x in shap_df['Impact']]
fig, ax = plt.subplots(figsize=(9, 4))
fig.patch.set_facecolor('#131315')
ax.set_facecolor('#131315')
ax.barh(shap_df['Feature'], shap_df['Impact'], color=colors, height=0.55)
ax.axvline(x=0, color='#262629', linewidth=1.5)
ax.set_xlabel('Impact on Churn Prediction', color='#8a8a93', fontsize=11)
ax.tick_params(colors='#8a8a93', labelsize=10)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.tight_layout()
st.pyplot(fig)

# ── Section 3: What-If ────────────────────────────────────
col_a, col_b = st.columns(2)
with col_a:
    whatif_active   = st.selectbox("Set Member Status", ["Yes", "No"],
                                    index=0 if is_active == "No" else 1, key="whatif_active")
with col_b:
    whatif_products = st.selectbox("Set Number of Products", [1, 2, 3, 4], key="whatif_products")

whatif_active_val      = 1 if whatif_active == "Yes" else 0
whatif_high_risk       = 1 if (whatif_products >= 3 and whatif_active_val == 0) else 0
whatif_inactive_senior = 1 if (age > 60 and whatif_active_val == 0) else 0

whatif_df = pd.DataFrame([{
    'CreditScore': credit_score, 'Geography': geography, 'Gender': gender_val,
    'Age': age, 'Tenure': tenure, 'Balance': balance, 'NumOfProducts': whatif_products,
    'IsActiveMember': whatif_active_val, 'IsZeroBalance': is_zero_balance, 'AgeGroup': age_group,
    'IsHighRiskProduct': whatif_high_risk, 'IsGermany': is_germany,
    'BalanceToSalaryRatio': balance_salary_ratio, 'IsInactiveSenior': whatif_inactive_senior,
    'IsFemaleGermany': is_female_germany
}])

whatif_prob = model.predict_proba(whatif_df)[0][1]
delta       = whatif_prob - prob

if whatif_prob < 0.3:
    after_color, after_label = "#22c55e", "Low Risk"
elif whatif_prob < 0.5:
    after_color, after_label = "#eab308", "Medium Risk"
elif whatif_prob < 0.7:
    after_color, after_label = "#f97316", "High Risk"
else:
    after_color, after_label = "#ef4444", "Critical Risk"

delta_color = "#22c55e" if delta < 0 else "#ef4444"
delta_sign  = "+" if delta > 0 else ""

st.markdown(f"""
<div style="background:#131315; border:1px solid #262629; border-radius:12px; padding:22px 24px; margin-top:8px;">
    <div style="padding-bottom:16px; border-bottom:1px solid #262629; margin-bottom:20px;">
        <h2 style="font-size:13px; font-weight:500; color:#f4f4f5; margin:0;">What-If Analysis</h2>
        <p style="font-size:12px; color:#8a8a93; margin-top:2px;">Simulate changes to estimate impact on churn risk</p>
    </div>
    <div style="display:grid; grid-template-columns:1fr auto 1fr; align-items:stretch; gap:14px; margin-top:8px;">
        <div style="background:#0a0a0a; border:1px solid #262629; border-radius:10px; padding:18px; text-align:center;">
            <p style="font-size:11px; font-weight:500; color:#8a8a93; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:10px;">Before</p>
            <p style="font-size:36px; font-weight:700; line-height:1; color:{text_color};">{prob*100:.0f}%</p>
            <p style="font-size:12px; color:#8a8a93; margin-top:8px;">{risk_label}</p>
        </div>
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; gap:4px; padding:0 8px;">
            <span style="font-size:18px; color:#8a8a93;">&rarr;</span>
            <span style="font-size:14px; font-weight:700; color:{delta_color};">{delta_sign}{abs(delta)*100:.0f}%</span>
            <span style="font-size:11px; color:#8a8a93; text-transform:uppercase; letter-spacing:0.05em;">change</span>
        </div>
        <div style="background:#0a0a0a; border:1px solid #262629; border-radius:10px; padding:18px; text-align:center;">
            <p style="font-size:11px; font-weight:500; color:#8a8a93; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:10px;">After</p>
            <p style="font-size:36px; font-weight:700; line-height:1; color:{after_color};">{whatif_prob*100:.0f}%</p>
            <p style="font-size:12px; color:#8a8a93; margin-top:8px;">{after_label}</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)