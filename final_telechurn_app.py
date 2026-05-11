"""
Telecom Customer Churn — Production ML Pipeline & ROI System
=====================================================
✅ Pipeline Native Prediction & Leakage Safe
✅ ADVANCED GRAPHICS: Dark Mode, Gridlines, Vibrant Colors
✅ HIGHLY READABLE: Step-by-step hashtag commented plotting code
✅ MEMORY SAFE: Sparse categorical outputs, correct SMOTE usage
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
    precision_score, recall_score,
    roc_curve, auc, precision_recall_curve, roc_auc_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────────────────────────────────────
# 0. GLOBAL UI & GRAPHICS SETUP (Black Background, White Text, Grids)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Telecom Churn ML System", layout="wide")

# Set Matplotlib Global Dark Theme
plt.style.use('dark_background')
plt.rcParams.update({
    "axes.facecolor": "black",       # Black background for the actual plot area
    "figure.facecolor": "black",     # Black background for the outer figure
    "text.color": "white",           # White text
    "axes.labelcolor": "white",      # White axis labels
    "xtick.color": "white",          # White X ticks
    "ytick.color": "white",          # White Y ticks
    "grid.color": "#444444",         # Dark gray grid lines
    "grid.linestyle": "--",          # Dashed grid lines
    "grid.alpha": 0.7                # Slightly transparent grid lines
})

# Neon Color Palette for our graphs
COLOR_NO  = '#00FFFF' # Cyan (For 'No Churn' / Good things)
COLOR_YES = '#FF00FF' # Magenta (For 'Churn' / Bad things)
COLOR_ALT = '#39FF14' # Neon Green (For neutral/highlight things)

st.title(" Telecom Customer Churn — Production ML System")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING FACTORY
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.subheader("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Telco_Customer_Churn.xlsx/.csv", type=["xlsx", "csv"])

if uploaded_file is None:
    st.warning("⚠️ Please upload the dataset in the sidebar to proceed.")
    st.stop()

@st.cache_data
def load_data(file):
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)
        
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    
    # Encode target to binary explicitly
    if "Churn" in df.columns and df["Churn"].dtype == object:
        df["Churn"] = (df["Churn"] == "Yes").astype(int)
        
    return df

df = load_data(uploaded_file)

TARGET = "Churn"
X_raw = df.drop(TARGET, axis=1)
y = df[TARGET]

num_cols = X_raw.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_raw.select_dtypes(include="object").columns.tolist()

def make_preprocessor(include_scaler: bool) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if include_scaler:
        num_steps.append(("scaler", StandardScaler()))
        
    cat_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),  
        # FIXED: Sparse output true by default. Prevents memory explosion.
        ("ohe", OneHotEncoder(handle_unknown="ignore")), 
    ])

    return ColumnTransformer([
        ("num", SkPipeline(num_steps), num_cols),  
        ("cat", cat_pipe, cat_cols),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# 2. SIDEBAR CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    test_size = st.slider("Test Size", 0.05, 0.3, 0.20, 0.01)
    random_state = st.number_input("Random State", 0, 100, 42)

    st.divider()
    st.subheader("Prediction Threshold")
    threshold = st.slider("Churn Probability Threshold", 0.0, 1.0, 0.58, 0.01)

    st.divider()
    st.subheader("Logistic Regression")
    lr_c        = st.slider("LR — C (Regularization)", 0.001, 10.0, 0.1, 0.001)
    lr_max_iter = st.slider("LR — Max Iterations", 1000, 10000, 5000, 100)
    lr_solver   = st.selectbox("LR — Solver", ["lbfgs", "liblinear", "saga"])

    st.divider()
    st.subheader("Random Forest")
    rf_n_estimators      = st.slider("RF — Trees", 10, 500, 190, 10)
    rf_max_depth         = st.slider("RF — Max Depth", 1, 30, 8, 1)
    rf_bootstrap         = st.selectbox("RF — Bootstrap", [True, False])

    st.divider()
    st.subheader("LightGBM")
    lgb_n_estimators  = st.slider("LGB — Trees", 50, 500, 300, 10)
    lgb_learning_rate = st.slider("LGB — Learning Rate", 0.001, 0.5, 0.05, 0.001)
    lgb_max_depth     = st.slider("LGB — Max Depth", 3, 15, 7, 1)
    lgb_num_leaves    = st.slider("LGB — Num Leaves", 20, 100, 31, 5)

    st.divider()
    
    st.subheader("Business ROI Inputs")
    avg_revenue_per_customer = st.number_input("Avg Monthly Rev / Customer (₹)", 10, 500, 65)
    avg_retention_cost = st.number_input("Avg Retention Cost / Customer (₹)", 1, 200, 20)
    avg_acquisition_cost = st.number_input("Avg Acquisition Cost (₹)", 50, 2000, 350)
    churn_reduction_pct = st.slider("Expected Churn Reduction with ML (%)", 0, 100, 30)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINING & EVALUATION PIPELINE (Leakage Free)
# ─────────────────────────────────────────────────────────────────────────────
rs = int(random_state)
# Use a standard Train vs Test split since threshold tuning is now interactive via slider
X_trainval, X_test, y_trainval, y_test = train_test_split(X_raw, y, test_size=test_size, random_state=rs, stratify=y)

# =============================================================================
# PIPELINE DEFINITIONS WITH SLIDER VALUES
# =============================================================================
pipelines = {
    "Logistic Regression": ImbPipeline([
        ("pre", make_preprocessor(include_scaler=True)),  
        ("smote", SMOTE(random_state=rs)),  
        ("clf", LogisticRegression(
            C=lr_c, 
            solver=lr_solver, 
            max_iter=lr_max_iter, 
            class_weight="balanced", 
            random_state=rs
        )),
    ]),
    "Random Forest": ImbPipeline([
        ("pre", make_preprocessor(include_scaler=False)),  
        ("smote", SMOTE(random_state=rs)),  
        ("clf", RandomForestClassifier(
            n_estimators=rf_n_estimators, 
            max_depth=rf_max_depth, 
            bootstrap=rf_bootstrap,
            class_weight="balanced",
            random_state=rs
        )),
    ]),
    "LightGBM": SkPipeline([ 
        ("pre", make_preprocessor(include_scaler=False)),  
        # LGBM handles imbalance natively
        ("clf", LGBMClassifier(
            n_estimators=lgb_n_estimators, 
            learning_rate=lgb_learning_rate, 
            max_depth=lgb_max_depth, 
            num_leaves=lgb_num_leaves,
            class_weight="balanced", 
            random_state=rs, 
        )),
    ]),
}


results = []
best_pipe = None
best_f1 = 0
best_name = ""

with st.spinner("Training models with selected hyperparameters..."):
    for name, pipe in pipelines.items():
        pipe.fit(X_trainval, y_trainval)
        
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred_temp = (y_prob >= threshold).astype(int)
        
        # Calculating all the new binary metrics needed for the UI
        acc = accuracy_score(y_test, y_pred_temp)
        f1_churn = f1_score(y_test, y_pred_temp)
        rec_churn = recall_score(y_test, y_pred_temp)
        prec_churn = precision_score(y_test, y_pred_temp)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results.append([name, roc_auc, f1_churn, rec_churn, prec_churn, acc, threshold])
        
        # Tracking best model based on F1 Churn
        if f1_churn > best_f1:
            best_f1 = f1_churn
            best_pipe = pipe
            best_name = name
        results_df = pd.DataFrame(results, columns=[
        "Model", "ROC-AUC", "F1 Score (Churn)", "Recall (Churn)", 
        "Precision (Churn)", "Accuracy", "Threshold"
        ])

# ─────────────────────────────────────────────────────────────────────────────
# MANUAL MODEL SELECTION OVERRIDE
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("Select Model for Downstream Analysis")

model_options = ["Use Auto-Selected Best Model"] + list(pipelines.keys())

# FIX 1: Added a unique 'key' to force Streamlit to remember the selection
selected_model_choice = st.selectbox(
    "Choose which model to use for feature importance and evaluation:", 
    model_options,
    key="model_selector_override"
)

# Assign the pipeline and name based on the dropdown
if selected_model_choice == "Use Auto-Selected Best Model":
    model_choice = best_name
    final_pipe = best_pipe
    final_name = best_name
    st.info(f"Currently analyzing: **{final_name}** (Auto-Selected)")
else:
    model_choice = selected_model_choice
    final_pipe = pipelines[selected_model_choice]
    final_name = selected_model_choice
    st.info(f"Currently analyzing: **{final_name}** (Manually Selected)")

# FIX 2: Generate the predictions using final_pipe immediately after selection
# These variables MUST be the ones you use in your Tab 4 Evaluation metrics!
y_prob_test = final_pipe.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= threshold).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE IMPORTANCE EXTRACTION (Using Selected Model)
# ─────────────────────────────────────────────────────────────────────────────
if final_pipe is not None:
    clf_step = final_pipe.named_steps["clf"]
    pre_step = final_pipe.named_steps["pre"]
    
    # Extract feature names safely
    ohe_names = pre_step.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + ohe_names

    # Tree models use feature_importances_, LR uses absolute coefficients.
    importances = clf_step.feature_importances_ if hasattr(clf_step, "feature_importances_") else np.abs(clf_step.coef_[0])
    
    feat_df = pd.DataFrame({
        "Feature": all_feature_names, 
        "Importance": importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    def encoded_to_original(feat):
        for col in X_raw.columns:
            if feat == col or feat.startswith(col + "_"): 
                return col
        return None

    top10 = feat_df.head(10).reset_index(drop=True)
    top6, seen = [], set()
    
    for f in feat_df["Feature"].tolist():
        orig = encoded_to_original(f) or f
        if orig not in seen:
            top6.append(orig)
            seen.add(orig)
        if len(top6) == 6: 
            break

# ─────────────────────────────────────────────────────────────────────────────
# 5. TABS SETUP
# ─────────────────────────────────────────────────────────────────────────────
tab_data, tab_eda, tab_cmp, tab_eval, tab_feat, tab_roi, tab_pred, tab_strats = st.tabs([
   "Data", "EDA", " Model Comparison", "Evaluation",
    " Feature Insights", " Business ROI",
    " Predict & Risk", " Strategies"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DATA
# ─────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.subheader(" Telecom Customer Churn Dataset")
    st.write(df.head())

    st.divider()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Data Types**")
        st.write(df.dtypes.rename("dtype").reset_index()
                 .rename(columns={"index": "Column"}))
    with col_b:
        st.markdown("**Missing Values**")
        miss = df.isnull().sum().reset_index()
        miss.columns = ["Column", "Missing"]
        miss = miss[miss["Missing"] > 0]
        st.write(miss if len(miss) > 0
                 else pd.DataFrame({"Status": ["✅ No missing values"]}))
    with col_c:
        st.markdown("**Basic Statistics**")
        st.write(df.describe().T)

    st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — EDA (ADVANCED GRAPHICS)
# ─────────────────────────────────────────────────────────────────────────────
with tab_eda:
    st.subheader("🔍 Exploratory Data Analysis")
    
    # Ensure global colors are defined if not already at the top of your script
    COLOR_NO  = '#00FFFF' # Cyan (No Churn)
    COLOR_YES = '#FF00FF' # Magenta (Churn)
    COLOR_ALT = '#39FF14' # Neon Green (Accents)

    # Create a display copy so we don't break the ML pipeline's binary 0/1 labels
    df_eda = df.copy()
    df_eda["Churn"] = df_eda["Churn"].map({1: "Yes", 0: "No"})

    # ── 1. Churn Distribution ─────────────────────────────────────────────────
    st.markdown("Churn Distribution")
    churn_counts = df_eda["Churn"].value_counts()
    churn_pct    = df_eda["Churn"].value_counts(normalize=True) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{len(df_eda):,}")
    col2.metric("Churned",  f"{churn_counts.get('Yes', 0):,} ({churn_pct.get('Yes', 0):.1f}%)")
    col3.metric("Retained", f"{churn_counts.get('No',  0):,} ({churn_pct.get('No',  0):.1f}%)")

    col1, col2 = st.columns(2)
    with col1:
        # ---------------------------------------------------------
        # BAR CHART: Overall Churn Count
        # ---------------------------------------------------------
        # 1. Create figure
        fig, ax = plt.subplots(figsize=(6, 4))
        # 2. Plot bars with specific neon colors
        bars = ax.bar(churn_counts.index, churn_counts.values, color=[COLOR_NO, COLOR_YES])
        # 3. Add Gridlines for readability
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        # 4. Add Labels
        ax.set_title("Churn Count Bar Chart")
        ax.set_ylabel("Number of Customers")
        # 5. Add Text annotations on top of bars
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30, 
                    f"{int(bar.get_height())}", ha="center", color="white", fontweight="bold")
        st.pyplot(fig)
        
    with col2:
        # ---------------------------------------------------------
        # PIE CHART: Overall Churn Share
        # ---------------------------------------------------------
        # 1. Create figure
        fig, ax = plt.subplots(figsize=(6, 4))
        # 2. Plot pie chart with white text props
        ax.pie(churn_counts.values, labels=churn_counts.index, autopct="%1.1f%%", 
               startangle=90, colors=[COLOR_NO, COLOR_YES], textprops={'color':"white", 'weight':'bold'})
        # 3. Add Title
        ax.set_title("Churn Share (%)")
        st.pyplot(fig)
    
    st.info(f" **Insight:** The baseline churn rate for this dataset is **{churn_pct.get('Yes', 0):.1f}%**. Any specific category exceeding this percentage represents a high-risk segment.")
    st.divider()

    # ── 2. Numeric Feature Distributions ──────────────────────────────────────
    st.markdown("Numeric Feature Distributions by Churn")
    num_features = df_eda.select_dtypes(include=np.number).columns.tolist()

    for num_col in num_features:
        st.markdown(f"{num_col}")
        churn_yes_n = df_eda[df_eda["Churn"] == "Yes"][num_col].dropna()
        churn_no_n  = df_eda[df_eda["Churn"] == "No"][num_col].dropna()

        col1, col2 = st.columns(2)
        with col1:
            # ---------------------------------------------------------
            # HISTOGRAM: Overlapping Distributions
            # ---------------------------------------------------------
            # 1. Create figure
            fig, ax = plt.subplots(figsize=(6, 4))
            # 2. Plot Stayed (Cyan) and Churned (Magenta) with transparency (alpha)
            ax.hist(churn_no_n,  bins=30, alpha=0.6, color=COLOR_NO,  label="Stayed (No)")
            ax.hist(churn_yes_n, bins=30, alpha=0.6, color=COLOR_YES, label="Churned (Yes)")
            # 3. Add Grid and Labels
            ax.grid(linestyle='--', alpha=0.5)
            ax.set_title(f"Histogram: {num_col}")
            ax.set_xlabel(num_col)
            ax.set_ylabel("Count")
            ax.legend()
            st.pyplot(fig)
            
        with col2:
            # ---------------------------------------------------------
            # LINE + SCATTER PLOT: Trend Analysis
            # ---------------------------------------------------------
            # 1. Create figure
            fig, ax = plt.subplots(figsize=(6, 4))
            # 2. Calculate continuous bins manually
            counts_yes, bins = np.histogram(churn_yes_n, bins=20)
            counts_no,  _    = np.histogram(churn_no_n,  bins=bins)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # 3. Draw the continuous Lines
            ax.plot(bin_centers, counts_no,  color=COLOR_NO,  linewidth=2, label="Stayed (No)")
            ax.plot(bin_centers, counts_yes, color=COLOR_YES, linewidth=2, label="Churned (Yes)")
            
            # 4. Overlay the Scatter Points directly on the line
            ax.scatter(bin_centers, counts_no,  color="white", zorder=5)
            ax.scatter(bin_centers, counts_yes, color="white", zorder=5)
            
            # 5. Format and display
            ax.grid(linestyle='--', alpha=0.5)
            ax.set_title(f"Line & Scatter: {num_col} Trend")
            ax.set_xlabel(num_col)
            ax.set_ylabel("Density / Count")
            ax.legend()
            st.pyplot(fig)
    
        # DYNAMIC INSIGHT BLOCK FOR NUMERICS
        mean_yes = churn_yes_n.mean()
        mean_no = churn_no_n.mean()
        if mean_yes > mean_no:
            st.info(f" **Insight:** Customers who churned tend to have **higher {num_col}** on average (₹{mean_yes:.1f} vs ₹{mean_no:.1f}). High values here indicate increased risk.")
        else:
            st.info(f" **Insight:** Customers who churned tend to have **lower {num_col}** on average ({mean_yes:.1f} vs {mean_no:.1f}). For example, newer customers with lower tenure leave more frequently.")
    st.divider()

    # ── 3. Categorical Features — Churn Rate ─────────────────────────────────
    st.markdown("Categorical Features — Churn Rate (%)")
    cat_features = [c for c in df_eda.columns if df_eda[c].dtype == "object" and c != "Churn"]

    for cat in cat_features:
        cr = (df_eda.groupby(cat)["Churn"]
              .apply(lambda x: round((x == "Yes").sum() / len(x) * 100, 2))
              .reset_index())
        cr.columns    = [cat, "Churn Rate (%)"]
        categories_c  = cr[cat].tolist()
        rates_c       = cr["Churn Rate (%)"].tolist()

        st.markdown(f"{cat} vs Churn")
        col1, col2 = st.columns(2)
        
        with col1:
            # ---------------------------------------------------------
            # BAR CHART: Categorical Churn Rate
            # ---------------------------------------------------------
            fig, ax = plt.subplots(figsize=(6, 4))
            # Plot with Magenta to signify risk
            bars = ax.bar(categories_c, rates_c, color=COLOR_YES)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.set_ylabel("Churn Rate (%)")
            ax.set_title(f"Bar Chart: Churn Rate by {cat}")
            plt.xticks(rotation=15)
            # Add percentage text on top
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f"{bar.get_height()}%", ha="center", color="white", fontsize=9)
            st.pyplot(fig)
            
        with col2:
            # ---------------------------------------------------------
            # PIE CHART: Categorical Share
            # ---------------------------------------------------------
            fig, ax = plt.subplots(figsize=(6, 4))
            # Generate a dynamic cool color palette based on number of categories
            pie_colors = sns.color_palette("cool", len(categories_c))
            ax.pie(rates_c, labels=categories_c, autopct="%1.1f%%", colors=pie_colors, textprops={'color':"white"})
            ax.set_title(f"Pie Chart: {cat} Churn Share")
            st.pyplot(fig)
    
        # DYNAMIC INSIGHT BLOCK FOR CATEGORICALS
        highest_risk_category = cr.loc[cr["Churn Rate (%)"].idxmax(), cat]
        highest_rate = cr["Churn Rate (%)"].max()
        st.info(f" **Insight:** Within **{cat}**, the **'{highest_risk_category}'** segment carries the highest churn risk at **{highest_rate:.1f}%**.")
        st.divider()

    
    # ── 4. Tenure Segmentation ────────────────────────────────────────────────
    if "tenure" in df_eda.columns:
        st.markdown("###  Tenure Segmentation vs Churn")
        df_seg = df_eda.copy()
        df_seg["Tenure Segment"] = pd.cut(
            df_seg["tenure"],
            bins=[0, 12, 24, 36, 48, 60, 72],
            labels=["0-12m", "13-24m", "25-36m", "37-48m", "49-60m", "61-72m"],
        )
        seg_churn = (
            df_seg.groupby("Tenure Segment", observed=True)["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index()
        )
        seg_churn.columns = ["Tenure Segment", "Churn Rate (%)"]

        col1, col2 = st.columns(2)
        with col1:
            # ---------------------------------------------------------
            # BAR CHART: Tenure Segments
            # ---------------------------------------------------------
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(seg_churn["Tenure Segment"].astype(str), seg_churn["Churn Rate (%)"], color=COLOR_ALT)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.set_title("Churn Rate by Tenure Segment")
            ax.set_ylabel("Churn Rate (%)")
            ax.set_xlabel("Tenure (months)")
            plt.xticks(rotation=15)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f"{bar.get_height():.1f}%", ha="center", color="white", fontsize=9)
            st.pyplot(fig)
            
        with col2:
            # ---------------------------------------------------------
            # PIE CHART: Tenure Segments
            # ---------------------------------------------------------
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(seg_churn["Churn Rate (%)"], labels=seg_churn["Tenure Segment"].astype(str),
                   autopct="%1.1f%%", colors=sns.color_palette("viridis", len(seg_churn)), textprops={'color':"white"})
            ax.set_title("Tenure Segment Churn Share")
            st.pyplot(fig)
    
        # SPECIFIC INSIGHT FOR TENURE BINS
        highest_bin = seg_churn.loc[seg_churn["Churn Rate (%)"].idxmax(), "Tenure Segment"]
        st.info(f" **Insight:** The **{highest_bin}** group is the most dangerous period in the customer lifecycle. Retention efforts should be heavily focused here before they mature.")

    # ── 5. Correlation Heatmap ────────────────────────────────────────────────
    st.markdown("Correlation Heatmap (Encoded Features)")
    # Generate temporary encoded dataframe strictly for the correlation heatmap
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # 1. Create a large figure to fit the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    # 2. Mask the upper triangle for cleaner viewing
    mask = np.triu(np.ones_like(df_encoded.corr(), dtype=bool))
    # 3. Plot with a dark-friendly colormap (mako or magma work well on black)
    sns.heatmap(df_encoded.corr(), mask=mask, annot=False,
                cmap="mako", ax=ax, linewidths=0.3, linecolor='black')
    ax.set_title("Feature Correlation Matrix", color="white", size=14)
    # Ensure tick labels are white
    ax.tick_params(colors='white')
    st.pyplot(fig)

    # DYNAMIC INSIGHT BLOCK FOR CORRELATION HEATMAP
    corr_matrix = df_encoded.corr()
    if "Churn" in corr_matrix.columns:
        # Sort absolute correlations with Churn (excluding Churn itself)
        churn_corr = corr_matrix["Churn"].drop("Churn").abs().sort_values(ascending=False)
        top_feature = churn_corr.index[0]
        top_val = corr_matrix.loc[top_feature, "Churn"]

        # Determine direction and business translation
        direction = "positive" if top_val > 0 else "negative"
        effect = "increases" if top_val > 0 else "decreases"
            
        st.info(f" **Insight:** The heatmap scans for mathematical relationships. It found that **{top_feature}** has the strongest {direction} correlation (**{top_val:.2f}**) with Churn. In business terms: as {top_feature} goes up, the likelihood of a customer leaving **{effect}**.")
    st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab_cmp:
    st.subheader("Model Comparison")
    
    # FORMATTER UPDATED
    format_dict = {
        "ROC-AUC": "{:.4f}",
        "F1 Score (Churn)": "{:.4f}", 
        "Recall (Churn)": "{:.4f}", 
        "Precision (Churn)": "{:.4f}", 
        "Accuracy": "{:.4f}", 
        "Threshold": "{:.2f}"
    }
    st.dataframe(results_df.style.format(format_dict), use_container_width=True)
    
    # CORRECTED: Changed "F1 Score" to "F1 Score (Churn)" to prevent KeyError
    best_row = results_df.loc[results_df["F1 Score (Churn)"].idxmax()]
    best_auc = best_row["ROC-AUC"] if "ROC-AUC" in best_row else 0.0 
    
    if selected_model_choice == "Use Auto-Selected Best Model":
        st.success(f" Currently Evaluating: **{final_name}** (Auto-Selected Best Model)")
    else:
        st.info(f" Currently Evaluating: **{final_name}** (Manually Selected Override)")
        
    col1, col2 = st.columns(2)
    with col1:
        # BAR CHART: ROC-AUC
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(results_df["Model"], results_df["ROC-AUC"], color=COLOR_ALT)
        ax.grid(axis='y')
        ax.set_title("ROC-AUC Comparison")
        ax.set_ylim(0, 1) 
        plt.xticks(rotation=15)
        st.pyplot(fig)
        
    with col2:
        # BAR CHART: Model F1 Score (Churn)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(results_df["Model"], results_df["F1 Score (Churn)"], color=COLOR_NO)
        ax.grid(axis='y')
        ax.set_title("F1 Score (Churn) Comparison")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=15)
        st.pyplot(fig)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — EVALUATION (Held Out Test Set)
# ─────────────────────────────────────────────────────────────────────────────
with tab_eval:
    st.subheader(f" Model Evaluation — { model_choice }")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric(" Accuracy",  f"{accuracy_score(y_test, y_pred_test):.4f}")
    col_b.metric(" Precision (Churn)", f"{precision_score(y_test, y_pred_test):.4f}")
    col_c.metric(" Recall (Churn)",    f"{recall_score(y_test, y_pred_test):.4f}")
    col_d.metric(" F1-Score (Churn)",  f"{f1_score(y_test, y_pred_test):.4f}")

    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred_test))

    st.divider()
    st.subheader(" Confusion Matrix & Metrics")
    cm = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

    col_roc1, col_roc2 = st.columns(2)
    with col_roc1:
        # LINE PLOT: ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob_test)
        roc_auc_val = auc(fpr, tpr) # Calculate safely 
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color=COLOR_YES, linewidth=2, label=f"AUC={roc_auc_val:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax.grid(True)
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

    with col_roc2:
        # LINE PLOT: Precision-Recall Curve
        prec_v, rec_v, _ = precision_recall_curve(y_test, y_prob_test)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rec_v, prec_v, color=COLOR_NO, linewidth=2)
        ax.grid(True)
        ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        st.pyplot(fig)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — FEATURE IMPORTANCE & INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_feat:
    st.subheader("  Feature Importance — Top 10 Drivers")
    
    # ---------------------------------------------------------
    # HORIZONTAL BAR CHART: Feature Importances
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    # Reverse the order so the largest is at the top
    y_pos = np.arange(len(top10))
    ax.barh(y_pos, top10["Importance"][::-1], color=COLOR_ALT)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top10["Feature"][::-1])
    ax.grid(axis='x')
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 10 Features (Highest → Lowest)")
    st.pyplot(fig)

    st.divider()
    st.markdown("###  Top 6 Features → Used for Business Insights")
    rank_table = pd.DataFrame({
        "Rank":    ["#1", "#2", "#3", "#4", "#5", "#6"],
        "Feature": top6,
    })
    st.table(rank_table)

    st.subheader("  Business Insights Based on Model's Top 6 Features")
    
    df_eda = df.copy()
    df_eda["Churn"] = df_eda["Churn"].map({1: "Yes", 0: "No"})
    
    for original_col in top6:
        if original_col not in df.columns or original_col == "Churn": continue
        st.markdown(f" {original_col} vs Churn")

        if df[original_col].dtype == "object":
            # Categorical Insights
            churn_rate = (df_eda.groupby(original_col)["Churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index())
            churn_rate.columns = [original_col, "Churn Rate (%)"]
            categories, rates = churn_rate[original_col].tolist(), churn_rate["Churn Rate (%)"].tolist()

            col1, col2 = st.columns(2)
            with col1:
                # ---------------------------------------------------------
                # BAR CHART: Categorical Churn Rate
                # ---------------------------------------------------------
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(categories, rates, color=COLOR_YES)
                ax.grid(axis='y')
                ax.set_ylabel("Churn Rate (%)")
                ax.set_title(f"Bar Chart: Churn Rate by {original_col}")
                plt.xticks(rotation=15)
                st.pyplot(fig)
            with col2:
                # ---------------------------------------------------------
                # PIE CHART: Categorical Churn Share
                # ---------------------------------------------------------
                fig, ax = plt.subplots(figsize=(6, 4))
                # Generate unique colors for pie slices dynamically
                pie_colors = sns.color_palette("cool", len(categories))
                ax.pie(rates, labels=categories, autopct='%1.1f%%', colors=pie_colors, textprops={'color':"white"})
                ax.set_title(f"Pie Chart: Churn Impact by {original_col}")
                st.pyplot(fig)
   
            # DYNAMIC INSIGHT BLOCK FOR ML CATEGORICALS
            highest_risk_category = churn_rate.loc[churn_rate["Churn Rate (%)"].idxmax(), original_col]
            highest_rate = churn_rate["Churn Rate (%)"].max()
            st.info(f" **Model Insight:** The ML model flagged **{original_col}** as a top driver of churn because the **'{highest_risk_category}'** segment is fleeing at a rate of **{highest_rate:.1f}%**. This is a critical risk factor.")
        
        else:
            # Numeric Insights
            churn_yes = df_eda[df_eda["Churn"] == "Yes"][original_col].dropna()
            churn_no  = df_eda[df_eda["Churn"] == "No"][original_col].dropna()

            col1, col2 = st.columns(2)
            with col1:
                # ---------------------------------------------------------
                # LINE PLOT WITH SCATTER: Trend Distribution
                # ---------------------------------------------------------
                fig, ax = plt.subplots(figsize=(6, 4))
                # 1. Calculate Histogram bins manually to plot as a Line
                counts_yes, bins = np.histogram(churn_yes, bins=20)
                counts_no,  _    = np.histogram(churn_no,  bins=bins)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                # 2. Draw the continuous Line
                ax.plot(bin_centers, counts_no,  color=COLOR_NO,  linewidth=2, label="Stayed (No)")
                ax.plot(bin_centers, counts_yes, color=COLOR_YES, linewidth=2, label="Churned (Yes)")
                
                # 3. Overlay the Scatter Points directly on the line
                ax.scatter(bin_centers, counts_no,  color="white", zorder=5)
                ax.scatter(bin_centers, counts_yes, color="white", zorder=5)
                
                ax.grid(True)
                ax.set_xlabel(original_col)
                ax.set_ylabel("Customer Count")
                ax.set_title("Line & Scatter: Distribution Trend")
                ax.legend()
                st.pyplot(fig)
                
            with col2:
                # ---------------------------------------------------------
                # SCATTER PLOT: Raw Data Scatter
                # ---------------------------------------------------------
                fig, ax = plt.subplots(figsize=(6, 4))
                # Plot every single customer point, separated by churn status
                ax.scatter(range(len(churn_no)),  churn_no,  color=COLOR_NO,  alpha=0.3, s=5, label="Stayed")
                ax.scatter(range(len(churn_yes)), churn_yes, color=COLOR_YES, alpha=0.3, s=5, label="Churned")
                ax.grid(True)
                ax.set_xlabel("Customer Index")
                ax.set_ylabel(original_col)
                ax.set_title("Raw Scatter Plot")
                ax.legend()
                st.pyplot(fig)
   
            # DYNAMIC INSIGHT BLOCK FOR ML NUMERICS
            mean_yes = churn_yes.mean()
            mean_no = churn_no.mean()
            if mean_yes > mean_no:
                st.info(f" **Model Insight:** The ML model heavily weights **{original_col}**. Notice that churned customers have significantly **higher** values on average (₹{mean_yes:.1f} vs ₹{mean_no:.1f}). When this metric goes up, the risk score spikes.")
            else:
                st.info(f" **Model Insight:** The ML model heavily weights **{original_col}**. Notice that churned customers have significantly **lower** values on average ({mean_yes:.1f} vs {mean_no:.1f}). A low value here is a massive red flag for impending churn.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — BUSINESS ROI SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
with tab_roi:
    st.subheader(" Business ROI — Churn Reduction Impact Analysis")
    
    total_customers   = len(df)
    current_churn_pct = y.mean()
    churned_customers = int(total_customers * current_churn_pct)
    annual_rev_loss   = churned_customers * avg_revenue_per_customer * 12

    customers_saved      = int(churned_customers * churn_reduction_pct / 100)
    retention_spend      = customers_saved * avg_retention_cost
    revenue_saved_annual = customers_saved * avg_revenue_per_customer * 12
    acq_cost_saved       = customers_saved * avg_acquisition_cost

    st.markdown("  Current Business State (Before ML)")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total Customers",     f"{total_customers:,}")
    r2.metric("Current Churn Rate",  f"{current_churn_pct*100:.1f}%")
    r3.metric("Churned / yr",        f"{churned_customers:,}")
    r4.metric("Annual Revenue Loss", f"₹{annual_rev_loss:,.0f}")
    
    st.markdown("  Business Revenue Impact")
    monthly_loss = churned_customers * avg_revenue_per_customer  
    annual_loss = monthly_loss * 12
    recoverable = annual_rev_loss - revenue_saved_annual

    b1, b2, b3 = st.columns(3)
    b1.metric("Monthly Revenue at Risk", f"₹{monthly_loss:,.0f}")
    b2.metric("Annual Revenue at Risk", f"₹{annual_loss:,.0f}")
    b3.metric("Recoverable via Retention", f"₹{recoverable:,.0f}")
    
    st.info("Check Tab 7 for the interactive Prediction and Visual Charts for Revenue/Churn reduction.")
    

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — PREDICT & ROI DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab_pred:
    st.subheader(" Predict Customer Churn")
    st.info(f" Focus on these Top 6 Features: **{', '.join(top6)}** to manage risk score.")
    
    cat_cols_pred = [c for c in X_raw.columns if X_raw[c].dtype == "object"]
    num_cols_pred = [c for c in X_raw.columns if X_raw[c].dtype != "object"]
    user_inputs   = {}

    st.markdown(" Customer Profile")
    priority_cat = ["gender", "SeniorCitizen", "Partner", "Dependents",
                    "PhoneService", "InternetService", "Contract",
                    "TechSupport", "OnlineSecurity", "PaymentMethod"]
    priority_num = ["tenure", "MonthlyCharges", "TotalCharges"]
    shown_cat, shown_num = [], []

    col1, col2, col3 = st.columns(3)
    cols_cycle = [col1, col2, col3]
    for i, field in enumerate(priority_cat):
        if field in cat_cols_pred:
            options = sorted(X_raw[field].dropna().unique().tolist())
            with cols_cycle[i % 3]:
                user_inputs[field] = st.selectbox(f"🔹 {field}", options, key=f"p_{field}")
            shown_cat.append(field)

    st.markdown("Usage & Charges")
    col4, col5, col6 = st.columns(3)
    cols_num = [col4, col5, col6]
    for i, field in enumerate(priority_num):
        if field in num_cols_pred:
            with cols_num[i % 3]:
                user_inputs[field] = st.number_input(
                    f"🔸 {field}",
                    float(X_raw[field].min()), float(X_raw[field].max()),
                    float(X_raw[field].mean()), key=f"p_{field}")
            shown_num.append(field)

    remaining_cat = [c for c in cat_cols_pred if c not in shown_cat]
    remaining_num = [c for c in num_cols_pred if c not in shown_num]
    if remaining_cat or remaining_num:
        with st.expander("➕ Additional Fields (optional — auto-filled with averages)"):
            for field in remaining_cat:
                options = sorted(X_raw[field].dropna().unique().tolist())
                user_inputs[field] = st.selectbox(f"{field}", options, key=f"r_{field}")
            for field in remaining_num:
                user_inputs[field] = st.number_input(
                    f"{field}",
                    float(X_raw[field].min()), float(X_raw[field].max()),
                    float(X_raw[field].mean()), key=f"r_{field}")

    # =========================================================================
    # THE PIPELINE FLEX: Build an aligned 1-row DataFrame mapping exactly to X_raw.
    # No pd.get_dummies, no reindexing, no manual scaler.transform!
    # =========================================================================
    input_df = pd.DataFrame([user_inputs])[X_raw.columns]

    st.divider()
    st.subheader(" Business Recommendation Prediction")

    # Simple risk scoring logic based on overall dataset
    churn_tables = {}
    for col in X_raw.columns:
        if X_raw[col].dtype == "object":
            churn_tables[col] = df.groupby(col)[TARGET].apply(lambda x: (x == 1).mean()).to_dict()

    risk_score, risk_factors = 0.0, []
    for feat in top6:
        original = None
        for col in X_raw.columns:
            if feat == col or feat.startswith(col + "_"):
                original = col
                break
        
        if original is None or original not in user_inputs:
            continue
            
        val = user_inputs[original]
        
        if original in churn_tables:
            if val in churn_tables[original]:
                cr = churn_tables[original][val]
                risk_score += cr
                lvl = "HIGH" if cr > 0.4 else ("MEDIUM" if cr > 0.2 else "LOW")
                risk_factors.append(f"{original}='{val}' → {lvl} churn ({cr:.1%})")
        else:
            mv = X_raw[original].mean()
            if float(val) > mv:
                risk_score += 0.3
                risk_factors.append(f"{original} above average chance of HIGH churn ({val:.1f})")
            else:
                risk_score += 0.1

    business_prob = min(risk_score / len(top6), 1.0)

    st.write("**Risk Factors Identified (Top Features):**")
    if risk_factors:
        for f in risk_factors:
            st.write(f"• {f}")
    else:
        st.write("✅ No major risk factors found.")
    st.write(f"**Business Risk Score: {business_prob:.2%}**")

    st.subheader("PREDICTION-DRIVEN BUSINESS ROI (BEFORE ML VS AFTER ML)")

    if st.button(" Predict Churn & ROI"):
        
        # --- CLEAN PIPELINE PREDICTION ---
        model_prob = final_pipe.predict_proba(input_df)[0][1]
        combined   = 0.7 * model_prob + 0.3 * business_prob

        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric(" Model Probability",   f"{model_prob:.2%}")
        col_s2.metric(" Business Risk Score",  f"{business_prob:.2%}")
        col_s3.metric(" Final Combined Score", f"{combined:.2%}")

        if combined > 0.6:
            st.error("🚨 Critical Risk — Immediate Action Needed!")
            st.write("→ Offer contract upgrade\n→ Provide free tech support\n→ Assign dedicated account manager")
        elif combined > 0.4:
            st.warning("⚠️ High Risk — Retention offer needed")
            st.write("→ Give loyalty discount (10-20%)\n→ Proactive outreach call")
        elif combined > 0.25:
            st.info("⚡ Medium Risk — Monitor closely")
            st.write("→ Send engagement email\n→ Offer service upgrade trial")
        else:
            st.success("✅ Low Risk — Maintain engagement")
            st.write("→ Standard newsletters and offers")

        # ======================================================================
        # Global ROI Math Based on Prediction
        # ======================================================================
        st.divider()
        st.subheader(" Business ROI — Based on This Prediction")

        total_cust          = len(df)
        baseline_rate       = y.mean() 
        effective_pred_rate = min(combined, baseline_rate)

        churners_before  = int(total_cust * baseline_rate)
        churners_after   = int(total_cust * effective_pred_rate)
        churners_reduced = churners_before - churners_after

        rev_lost_before  = churners_before  * avg_revenue_per_customer * 12
        rev_lost_after   = churners_after   * avg_revenue_per_customer * 12
        revenue_retained = churners_reduced * avg_revenue_per_customer * 12
        acq_cost_avoided = churners_reduced * avg_acquisition_cost
        ret_cost         = churners_reduced * avg_retention_cost
        
        net_roi_pred = revenue_retained + acq_cost_avoided - ret_cost
        roi_pct_pred = (net_roi_pred / ret_cost * 100) if ret_cost > 0 else 0
        churn_red_pp = (baseline_rate - effective_pred_rate) * 100

        st.markdown("Churn Rate — Before vs After ML")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Baseline Churn Rate", f"{baseline_rate*100:.1f}%")
        k2.metric("Predicted Churn Risk", f"{effective_pred_rate*100:.1f}%", delta=f"-{churn_red_pp:.1f} pp", delta_color="normal")
        k3.metric("Churners Before ML", f"{churners_before:,}")
        k4.metric("Churners After ML",  f"{churners_after:,}", delta=f"-{churners_reduced:,} reduced", delta_color="normal")

        st.divider()
        st.markdown("Revenue Impact — Before vs After ML")
        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Annual Rev Lost (Before)", f"₹{rev_lost_before:,.0f}")
        k6.metric("Annual Rev Lost (After)",  f"₹{rev_lost_after:,.0f}", delta=f"-₹{rev_lost_before - rev_lost_after:,.0f}", delta_color="normal")
        k7.metric("Revenue Retained by ML", f"₹{revenue_retained:,.0f}", delta=f"+₹{revenue_retained:,.0f}")
        k8.metric("Net ROI from ML", f"₹{net_roi_pred:,.0f}", delta=f"{roi_pct_pred:.0f}% ROI")

        st.divider()
        st.markdown("####  Visual Financial Impact: Before vs After ML")
        
        rev_col, churn_col = st.columns(2)

        with rev_col:
            # ---------------------------------------------------------
            # BAR CHART: Revenue Impact
            # ---------------------------------------------------------
            fig, ax = plt.subplots(figsize=(6, 4))
            labels = ["Lost (Before)", "Lost (After)", "Retained"]
            values = [rev_lost_before / 1e3, rev_lost_after / 1e3, revenue_retained / 1e3]
            
            # Map colors: Magenta for lost, Cyan for retained
            bar_colors = [COLOR_YES, '#FF6666', COLOR_NO]
            
            bars = ax.bar(labels, values, color=bar_colors)
            ax.grid(axis='y')
            ax.set_ylabel("Amount (₹ Thousands)")
            ax.set_title("Annual Revenue Impact")
            
            # Add text labels on top of bars
            for bar, val in zip(bars, [rev_lost_before, rev_lost_after, revenue_retained]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"₹{val/1e3:,.0f}K", ha="center", color="white")
            st.pyplot(fig)

        with churn_col:
            # ---------------------------------------------------------
            # BAR CHART: Churn Volume Impact
            # ---------------------------------------------------------
            fig, ax = plt.subplots(figsize=(6, 4))
            labels = ["Before ML", "After ML", "Reduced"]
            values = [churners_before, churners_after, churners_reduced]
            
            # Map colors
            bar_colors = [COLOR_YES, '#FF6666', COLOR_NO]
            
            bars = ax.bar(labels, values, color=bar_colors)
            ax.grid(axis='y')
            ax.set_ylabel("Number of Customers")
            ax.set_title("Total Churn Count Impact")
            
            # Add text labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(val), ha="center", color="white")
            st.pyplot(fig)

        # ── Full breakdown table — full width ──────────────────────────────────
        st.markdown("Full ROI Breakdown Table")
        breakdown = pd.DataFrame({
            "Metric": [
                "Total Customers in Dataset",
                "Baseline Churn Rate (Before ML)",
                "Predicted Churn Risk (After ML)",
                "Churn Rate Reduction",
                "Churners Before ML (population)",
                "Churners After ML (population)",
                "Churners Reduced (saved by ML)",
                "Annual Revenue Lost — Before ML",
                "Annual Revenue Lost — After ML",
                "Revenue Retained (saved)",
                "Acquisition Cost Avoided",
                "Retention Campaign Cost",
                "Net ROI",
                "ROI %",
            ],
            "Value": [
                f"{total_cust:,}",
                f"{baseline_rate*100:.1f}%",
                f"{effective_pred_rate*100:.1f}%",
                f"{churn_red_pp:.1f} percentage points",
                f"{churners_before:,}",
                f"{churners_after:,}",
                f"{churners_reduced:,}",
                f"Rs{rev_lost_before:,.0f}",
                f"Rs{rev_lost_after:,.0f}",
                f"Rs{revenue_retained:,.0f}",
                f"Rs{acq_cost_avoided:,.0f}",
                f"Rs{ret_cost:,.0f}",
                f"Rs{net_roi_pred:,.0f}",
                f"{roi_pct_pred:.1f}%",
            ],
        })
        st.table(breakdown)

        st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 — STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────
with tab_strats:
    st.header(" Churn Reduction Strategies and Revenue Recovery Plans")
    
    st.markdown("TABLE (Churn Reduction Strategies & Expected Impact)")
    churn_strategies = pd.DataFrame({
        "Strategy": ["Contract Flexibility", "Internet Speed/Reliability", "Reduce Charges", "Support Quality", "Loyalty Programs", "Issue Detection", "Service Recommendations"],
        "Expected Impact": ["↓ 8-10%", "↓ 5-7%", "↓ 3-5%", "↓ 4-6%", "↓ 6-8%", "↓ 7-9%", "↓ 5-7%"],
        "Timeline": ["2-3 mo", "3-4 mo", "1-2 mo", "Immed.", "1 mo", "2 mo", "1-2 mo"],
        "Revenue Impact": [f"₹{int(recoverable * 0.10):,}", f"₹{int(recoverable * 0.07):,}", f"₹{int(recoverable * 0.05):,}", f"₹{int(recoverable * 0.06):,}", f"₹{int(recoverable * 0.08):,}", f"₹{int(recoverable * 0.08):,}", f"₹{int(recoverable * 0.06):,}"]
    })
    st.dataframe(churn_strategies, use_container_width=True)
    
    st.markdown("---")
    st.markdown(f"###  Revenue Recovery Plan - ₹{int(recoverable):,} Target")
    recovery_plan = pd.DataFrame({
        "Recovery Strategy": ["Contract Upgrades", "Bundle Promos", "Retention Specialists", "Win-back Campaign", "VIP Program", "Long-term Discounts"],
        "Target": [f"{int(churned_customers * 0.25)} high-value", f"{int(churned_customers * 0.35)} medium", f"{int(churned_customers * 0.15)} top-tier", f"{int(churned_customers * 0.20)} churned", f"{int(churned_customers * 0.10)} loyal", f"{int(churned_customers * 0.40)} at-risk"],
        "Recovery": [f"₹{int(recoverable * 0.25):,}", f"₹{int(recoverable * 0.35):,}", f"₹{int(recoverable * 0.20):,}", f"₹{int(recoverable * 0.15):,}", f"₹{int(recoverable * 0.10):,}", f"₹{int(recoverable * 0.20):,}"]
    })
    st.dataframe(recovery_plan, use_container_width=True)
        
    st.markdown("---")
    st.markdown("###  Strategic Operational Recommendations")
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        st.markdown("**Operational Changes (Top Priority)**")
        op_actions = pd.DataFrame({
            "Action": ["Upgrade to 99.99% uptime", "Increase Internet speed +50Mbps", "Reduce outage resolution -80%", "24/7 multilingual support", "AI-powered chatbot"],
            "Impact": ["↓ 4-5%", "↓ 3-4%", "↓ 2-3%", "↓ 2-3%", "↓ 1-2%"],
            "Cost": ["₹500K", "₹300K", "₹150K", "₹400K", "₹200K"]
        })
        st.dataframe(op_actions, use_container_width=True)
    
    with rec_col2:
        st.markdown("**Quick Wins (Week 1)**")
        qw_actions = pd.DataFrame({
            "Action": ["Target 100 at-risk customers", "15% discount for 3 months", "Free service upgrade", "Instant callback system", "Email/SMS campaign"],
            "Impact": ["50 retained", "15-20% accept", "+25% satisfaction", "-50% wait time", "35% open rate"],
            "Revenue": ["₹50K/mo", "₹60K/mo", "₹30K/mo", "₹25K/mo", "₹40K/mo"]
        })
        st.dataframe(qw_actions, use_container_width=True)
