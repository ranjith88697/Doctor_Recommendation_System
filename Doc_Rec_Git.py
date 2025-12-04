# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from functools import lru_cache

st.set_page_config(page_title="Symptom → Disease → Doctor recommender", layout="wide")

st.title("Symptom-based Disease & Doctor Recommender")
st.write("Select symptoms, get a predicted disease, suggested specialist, and top recommended doctors (by Recommendation Score).")

# ---------------------------
# Utility: cached loaders
# ---------------------------
@st.cache_data
def load_original_dataset(path="Original_Dataset.csv"):
    return pd.read_csv(path)

@st.cache_data
def load_doc_vs_disease(path="Doctor_Versus_Disease.csv"):
    df = pd.read_csv(path, header=None, encoding='ISO-8859-1', engine='python')
    # create mapping dict: disease -> specialist
    try:
        mapping = df.set_index(0).to_dict()[1]
    except Exception:
        # fallback if header exists
        mapping = dict(zip(df.iloc[:,0].astype(str).str.strip(), df.iloc[:,1].astype(str).str.strip()))
    # normalize keys
    mapping = {str(k).strip(): str(v).strip() for k,v in mapping.items()}
    return mapping

@st.cache_data
def load_doctors(path="riga_doctors_sample.xlsx"):
    return pd.read_excel(path)

# ---------------------------
# Data preparation helper
# ---------------------------
def build_symptom_feature_matrix(original_df):
    """
    Based on the notebook logic: Original_Dataset rows contain Disease in first column and subsequent symptom columns.
    We'll build a binary symptom presence matrix (one-hot per symptom).
    """
    # Many dataset variants exist; try to detect columns that look like 'Symptom_' or treat row values as lists.
    df = original_df.copy()
    # If dataset has 'Disease' column, assume rest are symptoms columns or symptom names
    if 'Disease' in df.columns:
        # If dataset has many symptom columns with symptom names in each cell, we transform to binary
        # Collect unique symptom names from symptom columns
        symptom_cols = [c for c in df.columns if c.lower().startswith('symptom') or c not in ['Disease']]
        # If symptom_cols is empty, fallback: check all columns except first
        if not symptom_cols:
            symptom_cols = df.columns.tolist()
            symptom_cols.remove('Disease')
        # Create list of symptoms per row (ignore NaNs and empties)
        rows_symptoms = []
        for _, r in df[symptom_cols].iterrows():
            vals = [str(x).strip() for x in r.values if pd.notna(x) and str(x).strip() not in ['', '0', 'nan']]
            rows_symptoms.append(vals)
        mlb = MultiLabelBinarizer(sparse_output=False)
        X = pd.DataFrame(mlb.fit_transform(rows_symptoms), columns=mlb.classes_)
        X['Disease'] = df['Disease'].values
        return X, mlb
    else:
        # Another format: each row's non-null values except first are symptoms
        rows_symptoms = []
        diseases = []
        for _, r in df.iterrows():
            vals = r.dropna().astype(str).tolist()
            if len(vals) >= 1:
                disease = vals[0]
                symptoms = [v for v in vals[1:] if v not in ['', '0', 'nan']]
                diseases.append(disease)
                rows_symptoms.append(symptoms)
        mlb = MultiLabelBinarizer(sparse_output=False)
        X = pd.DataFrame(mlb.fit_transform(rows_symptoms), columns=mlb.classes_)
        X['Disease'] = diseases
        return X, mlb

# ---------------------------
# Train model (cached)
# ---------------------------
@st.cache_data
def train_model(X_df):
    """
    Train a RandomForest (same pipeline concept as notebook). Returns trained model and the training columns for constructing queries.
    X_df: DataFrame where last column is 'Disease' and rest are symptom binary columns.
    """
    X = X_df.drop(columns=['Disease'])
    y = X_df['Disease']
    # ensure columns are strings
    X.columns = X.columns.astype(str)
    # split (small portion just to allow measurement)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return clf, X.columns.tolist(), acc

# ---------------------------
# Recommendation Score helper
# ---------------------------
def compute_recommendation_score(doctors_df, weight_ratings=0.5, weight_satisfaction=0.5):
    df = doctors_df.copy()
    # Try to coerce numeric rating
    if 'Rating' in df.columns:
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    else:
        df['Rating'] = np.nan
    # Satisfaction may be like '85%' or numeric
    if 'Satisfaction' in df.columns:
        # strip % and cast
        df['Satisfaction'] = df['Satisfaction'].astype(str).str.rstrip('%').replace('nan','',regex=False)
        df['Satisfaction'] = pd.to_numeric(df['Satisfaction'], errors='coerce')/100.0
    else:
        df['Satisfaction'] = np.nan

    # Fill NaNs sensibly
    if df['Rating'].isna().all():
        df['Rating'] = 0.0
    if df['Satisfaction'].isna().all():
        df['Satisfaction'] = 0.0

    # Normalize by max to keep in 0-1
    max_rating = df['Rating'].max() if df['Rating'].max() > 0 else 1.0
    max_satisfaction = df['Satisfaction'].max() if df['Satisfaction'].max() > 0 else 1.0

    df['Normalized_Rating'] = df['Rating'] / max_rating
    df['Normalized_Satisfaction'] = df['Satisfaction'] / max_satisfaction

    df['Recommendation Score'] = (
        weight_ratings * df['Normalized_Rating'] + weight_satisfaction * df['Normalized_Satisfaction']
    )

    # Clip to 0-1 for safety
    df['Recommendation Score'] = df['Recommendation Score'].clip(0, 1)

    return df

# ---------------------------
# Load data
# ---------------------------
st.sidebar.header("Load data (files expected in same directory as this script)")
orig_path = st.sidebar.text_input("Original dataset CSV path", "Original_Dataset.csv")
mapping_path = st.sidebar.text_input("Doctor-vs-disease CSV path", "Doctor_Versus_Disease.csv")
doctors_path = st.sidebar.text_input("Doctors Excel path", "riga_doctors_sample.xlsx")

with st.spinner("Loading datasets..."):
    try:
        orig_df = load_original_dataset(orig_path)
    except Exception as e:
        st.error(f"Could not load Original_Dataset.csv from '{orig_path}': {e}")
        st.stop()
    try:
        doc_vs_disease_map = load_doc_vs_disease(mapping_path)
    except Exception as e:
        st.error(f"Could not load Doctor_Versus_Disease.csv from '{mapping_path}': {e}")
        st.stop()
    try:
        doctors_df = load_doctors(doctors_path)
    except Exception as e:
        st.error(f"Could not load doctors Excel from '{doctors_path}': {e}")
        st.stop()

# Build symptom feature matrix
with st.spinner("Preparing features and training model..."):
    X_df, mlb = build_symptom_feature_matrix(orig_df)
    model, feature_columns, model_acc = train_model(X_df)

# Sidebar show model accuracy
st.sidebar.markdown(f"**Model (RandomForest)** test accuracy: `{model_acc:.3f}`")
st.sidebar.markdown("If you want different behavior, change the classifier in the code.")

# ---------------------------
# Create user symptom input
# ---------------------------
st.header("Enter patient symptoms")
col1, col2 = st.columns([2,1])
with col1:
    all_symptoms = sorted([s for s in mlb.classes_])
    selected_symptoms = st.multiselect("Select symptoms (one or more)", all_symptoms, default=None)

with col2:
    st.markdown("**Controls**")
    top_n = st.number_input("Number of top doctors to show", min_value=1, max_value=20, value=5, step=1)
    weight_r = st.slider("Weight: Rating", 0.0, 1.0, 0.5)
    weight_s = 1.0 - weight_r
    st.markdown(f"Weight: Satisfaction = {weight_s:.2f}")

# Build query vector
def symptoms_to_vector(selected, feature_columns, mlb):
    # mlb.classes_ are symptoms; use transform
    if not selected:
        # empty => zero vector
        vec = np.zeros(len(feature_columns), dtype=int)
        return pd.DataFrame([vec], columns=feature_columns)
    onehot = mlb.transform([selected])
    # mlb.classes_ order may be different; ensure alignment to feature_columns
    oh_df = pd.DataFrame(onehot, columns=mlb.classes_)
    # reorder to feature_columns (some features could be missing if mismatch)
    for col in feature_columns:
        if col not in oh_df.columns:
            oh_df[col] = 0
    oh_df = oh_df[feature_columns]
    return oh_df

query_vec = symptoms_to_vector(selected_symptoms, feature_columns, mlb)

# Predict
if st.button("Predict disease & recommend doctors"):
    with st.spinner("Predicting..."):
        pred_disease = model.predict(query_vec)[0]
        st.success(f"Predicted disease: **{pred_disease}**")

        # Map to specialist
        recommended_specialist = doc_vs_disease_map.get(pred_disease, None)
        if recommended_specialist:
            st.info(f"Recommended specialist: **{recommended_specialist}**")
        else:
            st.warning("No specialist mapping found for this disease in `Doctor_Versus_Disease.csv`.")

        # Filter doctors by specialist (case-insensitive matching)
        df_docs = doctors_df.copy()
        if 'Specialty' in df_docs.columns and recommended_specialist:
            mask = df_docs['Specialty'].astype(str).str.strip().str.lower() == str(recommended_specialist).strip().lower()
            filtered = df_docs[mask].copy()
            if filtered.empty:
                # try partial match
                mask2 = df_docs['Specialty'].astype(str).str.strip().str.lower().str.contains(str(recommended_specialist).strip().lower())
                filtered = df_docs[mask2].copy()
        else:
            filtered = df_docs.copy()  # fallback: show all doctors

        if filtered.empty:
            st.warning("No doctors found for the recommended specialty. Showing all doctors as fallback.")
            filtered = df_docs.copy()

        # compute recommendation score
        scored = compute_recommendation_score(filtered, weight_ratings=weight_r, weight_satisfaction=weight_s)
        scored_sorted = scored.sort_values('Recommendation Score', ascending=False).reset_index(drop=True)

        # Display top N
        st.subheader(f"Top {top_n} recommended doctors")
        display_cols = [c for c in ['Name', 'Specialty', 'Clinic', 'Rating', 'Satisfaction', 'Recommendation Score'] if c in scored_sorted.columns]
        st.dataframe(scored_sorted[display_cols].head(top_n).assign(**{
            'Recommendation Score': lambda df: df['Recommendation Score'].round(3)
        }))

        # Also show a small bar chart for the top N
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, max(3, top_n*0.4)))
            top = scored_sorted.head(top_n)
            ax.barh(top['Name'].astype(str), top['Recommendation Score'])
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.set_xlabel("Recommendation Score")
            ax.set_title("Top doctors by Recommendation Score")
            st.pyplot(fig)
        except Exception:
            pass

        # Optional: show the full scored table in expanded view
        with st.expander("Show full filtered doctor list with scores"):
            st.dataframe(scored_sorted.assign(**{'Recommendation Score': scored_sorted['Recommendation Score'].round(4)}))
else:
    st.info("Choose symptoms and click **Predict disease & recommend doctors**.")

# Footer notes
st.markdown("---")
st.markdown("**Notes & assumptions**:")
st.markdown("""
- This app follows the same logic as your notebook: the Recommendation Score is a weighted sum of normalized `Rating` and `Satisfaction`.  
- If your doctors file has different column names, adjust the column names or normalize them before calling `compute_recommendation_score`.  
- The disease→specialist mapping expects `Doctor_Versus_Disease.csv` where first column is disease and second column is specialist (as in the notebook).  
- The disease classifier is trained on the provided `Original_Dataset.csv`. If the dataset format differs, modify `build_symptom_feature_matrix`.
""")
