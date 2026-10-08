
import io
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="MACHINE LEARNING-BASED PREDICTION OF DIABETES AND HEART DISEASE",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_STATE = 42

DIABETES_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
DIABETES_TARGET = "Outcome"

HEART_FEATURES = [
    "Age", "Sex", "ChestPain", "RestBP", "Chol", "FBS",
    "RestECG", "MaxHR", "ExAng", "Oldpeak", "Slope", "Ca", "Thal"
]
HEART_TARGET = "Target"

HEART_STANDARD_NAMES = HEART_FEATURES + [HEART_TARGET]

MODEL_ORDER = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "K-Nearest Neighbour",
    "Support Vector Machine"
]


def style_fig(fig):
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file)


def normalize_heart_columns(df):
    """Accept common Cleveland CSV variants and normalize to study names."""
    df = df.copy()

    # If exactly 14 unnamed columns exist, assign the study's schema.
    if df.shape[1] == 14 and not set(HEART_STANDARD_NAMES).issubset(df.columns):
        df.columns = HEART_STANDARD_NAMES
        return df

    rename_map = {}
    lower_to_actual = {str(c).strip().lower(): c for c in df.columns}

    aliases = {
        "age": "Age",
        "sex": "Sex",
        "cp": "ChestPain",
        "chestpain": "ChestPain",
        "chest pain": "ChestPain",
        "trestbps": "RestBP",
        "restbp": "RestBP",
        "restingbp": "RestBP",
        "chol": "Chol",
        "fbs": "FBS",
        "restecg": "RestECG",
        "thalach": "MaxHR",
        "maxhr": "MaxHR",
        "exang": "ExAng",
        "oldpeak": "Oldpeak",
        "slope": "Slope",
        "ca": "Ca",
        "thal": "Thal",
        "target": "Target",
        "num": "Target",
        "condition": "Target",
    }

    for lower, target in aliases.items():
        if lower in lower_to_actual:
            rename_map[lower_to_actual[lower]] = target

    df = df.rename(columns=rename_map)
    return df


def clean_diabetes(df):
    df = df.copy()

    missing_required = [c for c in DIABETES_FEATURES + [DIABETES_TARGET] if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Diabetes dataset is missing required columns: {missing_required}"
        )

    df = df[DIABETES_FEATURES + [DIABETES_TARGET]].copy()
    df = df.apply(pd.to_numeric, errors="coerce")

    zero_as_missing = [
        "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
    ]
    before = df[zero_as_missing].isna().sum().sum()
    df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)

    # Class-stratified median imputation, as specified in the dissertation.
    for col in zero_as_missing:
        df[col] = df.groupby(DIABETES_TARGET)[col].transform(
            lambda s: s.fillna(s.median())
        )
        # Fallback if a target class median is unavailable.
        df[col] = df[col].fillna(df[col].median())

    df[DIABETES_TARGET] = df[DIABETES_TARGET].astype(int)
    return df


def clean_heart(df):
    df = normalize_heart_columns(df)
    missing_required = [c for c in HEART_STANDARD_NAMES if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Heart dataset is missing required columns: {missing_required}"
        )

    df = df[HEART_STANDARD_NAMES].copy()
    df = df.replace("?", np.nan).apply(pd.to_numeric, errors="coerce")

    # Cleveland versions sometimes encode heart disease severity as 0-4.
    # Convert >0 to presence (1), matching the dissertation's binary target.
    df = df.dropna(subset=[HEART_TARGET]).copy()
    df[HEART_TARGET] = (df[HEART_TARGET] > 0).astype(int)

    for col in HEART_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def summary_table(df, target):
    tbl = df.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    return tbl.round(3)


def class_distribution(df, target, labels):
    counts = df[target].value_counts().sort_index()
    out = pd.DataFrame({
        "Class": counts.index.astype(int),
        "Class Label": [labels.get(int(i), str(i)) for i in counts.index],
        "Count": counts.values,
    })
    out["Percentage (%)"] = (out["Count"] / out["Count"].sum() * 100).round(1)
    return out


def interpretation(value):
    a = abs(value)
    if a >= 0.4:
        return "Strong positive" if value > 0 else "Strong negative"
    if a >= 0.2:
        return "Moderate positive" if value > 0 else "Moderate negative"
    if a >= 0.1:
        return "Weak positive" if value > 0 else "Weak negative"
    return "Very weak positive" if value > 0 else "Very weak negative"


def diabetes_corr_table(df):
    corr = df.corr(numeric_only=True)[DIABETES_TARGET].drop(DIABETES_TARGET)
    out = corr.sort_values(ascending=False).reset_index()
    out.columns = ["Feature", "Correlation with Outcome"]
    out["Correlation with Outcome"] = out["Correlation with Outcome"].round(3)
    out["Interpretation"] = out["Correlation with Outcome"].apply(interpretation)
    return out


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbour": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
    }


def build_param_grids():
    return {
        "Logistic Regression": {"C": [0.1, 1, 10, 100]},
        "Decision Tree": {
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_leaf": [1, 2, 5, 10],
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
        },
        "K-Nearest Neighbour": {"n_neighbors": [3, 5, 7, 9, 11]},
        "Support Vector Machine": {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto"],
            "kernel": ["rbf", "linear"],
        },
    }


@st.cache_data(show_spinner=False)
def run_full_experiment(X, y, use_smote=True, test_size=0.20):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = build_models()
    grids = build_param_grids()

    results = []
    trained = {}
    predictions = {}
    probabilities = {}
    cms = {}
    tuning_details = {}

    for name in MODEL_ORDER:
        model = models[name]
        grid = grids[name]

        steps = [("scaler", StandardScaler())]
        if use_smote:
            steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
        steps.append(("model", model))

        pipe = ImbPipeline(steps=steps)

        # Grid params must be prefixed for pipeline.
        prefixed_grid = {f"model__{k}": v for k, v in grid.items()}

        search = GridSearchCV(
            estimator=pipe,
            param_grid=prefixed_grid,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            refit=True
        )

        search.fit(X_train, y_train)
        best = search.best_estimator_

        pred = best.predict(X_test)
        prob = best.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        auc = roc_auc_score(y_test, prob)

        results.append({
            "Model": name,
            "Accuracy (%)": round(acc * 100, 2),
            "Precision (%)": round(prec * 100, 2),
            "Recall (%)": round(rec * 100, 2),
            "F1-Score (%)": round(f1 * 100, 2),
            "AUC-ROC": round(auc, 3),
        })

        trained[name] = best
        predictions[name] = pred
        probabilities[name] = prob
        cms[name] = confusion_matrix(y_test, pred)
        tuning_details[name] = {
            "Best Parameters": {
                k.replace("model__", ""): v for k, v in search.best_params_.items()
            },
            "Best CV F1": round(search.best_score_, 4)
        }

    result_df = pd.DataFrame(results)
    return {
        "results": result_df,
        "trained": trained,
        "predictions": predictions,
        "probabilities": probabilities,
        "cms": cms,
        "tuning": tuning_details,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "cv": cv
    }


def feature_importance_from_pipeline(best_pipe, feature_names):
    rf = best_pipe.named_steps["model"]
    vals = rf.feature_importances_
    return (
        pd.DataFrame({"Feature": feature_names, "Importance": vals})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def make_confusion_df(cm, labels):
    return pd.DataFrame(cm, index=[f"Actual {labels[0]}", f"Actual {labels[1]}"],
                        columns=[f"Predicted {labels[0]}", f"Predicted {labels[1]}"])


def make_download(df, filename, index=False):
    return st.download_button(
        f"Download {filename}",
        data=df.to_csv(index=index).encode("utf-8"),
        file_name=filename,
        mime="text/csv"
    )


# ------------------------------------ SIDEBAR -------------------------------------------------------------
st.sidebar.title("🩺 Four ML App")
st.sidebar.caption("Diabetes & Heart Disease Prediction and Analysis")
page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Data Upload",
        "Data Presentation",
        "Preprocessing",
        "Model Training",
        "Model Evaluation",
        "Interactive Prediction",
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("Random seed:", RANDOM_STATE)
st.sidebar.write("Train/test split:", "80% / 20%")
st.sidebar.write("Cross-validation:", "5-fold stratified")
st.sidebar.write("SMOTE:", "Training data only")


# ---------- Session state ----------
for key in ["diabetes_raw", "heart_raw", "diabetes", "heart", "diabetes_exp", "heart_exp"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ---------- Pages ----------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem; /* Adjust this value to move content higher up */
            padding-bottom: -1rem;
            
    </style>

    """,
    unsafe_allow_html=True
)

if page == "Dashboard":
    st.title("Welcome To Machine Learning-Based Prediction of Diabetes and Heart Disease", text_alignment="center")
    st.subheader("Data Presentation and Analysis" , text_alignment="center")

    st.image("dashboad_image_2.jpeg")

    st.write(
        "Interactive implementation of the analysis workflow. "
        "Upload the two CSV datasets to generate the tables, figures, models, and predictions dynamically."
    )


elif page == "Data Upload":
    st.title("Data Upload")
    st.write("Upload the exact CSV files used in your study.")

    d_file = st.file_uploader(
        "Upload Pima Indians Diabetes CSV",
        type=["csv"],
        key="d_upload"
    )

    h_file = st.file_uploader(
        "Upload Cleveland Heart Disease CSV",
        type=["csv"],
        key="h_upload"
    )

    if d_file is not None:
        st.session_state.diabetes_raw = load_csv(d_file)
        try:
            st.session_state.diabetes = clean_diabetes(st.session_state.diabetes_raw)
            st.success(f"Diabetes dataset loaded: {len(st.session_state.diabetes):,} records.")
        except Exception as e:
            st.error(str(e))

    if h_file is not None:
        st.session_state.heart_raw = load_csv(h_file)
        try:
            st.session_state.heart = clean_heart(st.session_state.heart_raw)
            st.success(f"Heart dataset loaded: {len(st.session_state.heart):,} records.")
        except Exception as e:
            st.error(str(e))

    st.markdown("### Expected Features")
    st.code(
        "Diabetes:\n"
        + ", ".join(DIABETES_FEATURES + [DIABETES_TARGET])
        + "\n\nHeart:\n"
        + ", ".join(HEART_STANDARD_NAMES)
    )

elif page == "Data Presentation":
    st.title("4.1 Presentation of Data")

    if st.session_state.diabetes is None or st.session_state.heart is None:
        st.warning("Upload both datasets on the Data Upload page first.")
        st.stop()

    d = st.session_state.diabetes
    h = st.session_state.heart

    tab1, tab2, tab3, tab4 = st.tabs([
        "Tables 4.1–4.2",
        "Tables 4.3–4.4",
        "Figures 4.1–4.2",
        "Table 4.5",
    ])

    with tab1:
        st.markdown("### Table 4.1: Summary Statistics of the Diabetes Dataset")
        t41 = summary_table(d, DIABETES_TARGET)
        st.dataframe(t41, use_container_width=True)
        make_download(t41, "Table_4_1_Diabetes_Summary.csv")

        st.markdown("### Table 4.2: Summary Statistics of the Heart Disease Dataset")
        t42 = summary_table(h, HEART_TARGET)
        st.dataframe(t42, use_container_width=True)
        make_download(t42, "Table_4_2_Heart_Summary.csv")

    with tab2:
        st.markdown("### Table 4.3: Class Distribution of Diabetes Dataset")
        t43 = class_distribution(
            d, DIABETES_TARGET, {0: "Non-Diabetic", 1: "Diabetic"}
        )
        st.dataframe(t43, use_container_width=True)

        st.markdown("### Table 4.4: Class Distribution of Heart Disease Dataset")
        t44 = class_distribution(
            h, HEART_TARGET, {0: "No Heart Disease", 1: "Heart Disease Present"}
        )
        st.dataframe(t44, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Figure 4.1: Distribution of BMI in Diabetes Dataset")
            fig1, ax1 = plt.subplots(figsize=(7, 4.5))
            ax1.hist(d["BMI"], bins=20, edgecolor="black")
            ax1.set_xlabel("BMI (kg/m²)")
            ax1.set_ylabel("Frequency")
            ax1.set_title("Distribution of BMI")
            st.pyplot(style_fig(fig1))
            plt.close(fig1)

        with c2:
            st.markdown("### Figure 4.2: Age Distribution in Heart Disease Dataset")
            fig2, ax2 = plt.subplots(figsize=(7, 4.5))
            ax2.hist(h["Age"], bins=15, edgecolor="black")
            ax2.set_xlabel("Age (Years)")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Age Distribution")
            st.pyplot(style_fig(fig2))
            plt.close(fig2)

    with tab4:
        st.markdown("### Table 4.5: Correlation of Features with Target Variable (Diabetes)")
        t45 = diabetes_corr_table(d)
        st.dataframe(t45, use_container_width=True)
        make_download(t45, "Table_4_5_Diabetes_Correlation.csv")

   

elif page == "Preprocessing":
    st.title("Data Preprocessing")

    if st.session_state.diabetes is None or st.session_state.heart is None:
        st.warning("Upload both datasets first.")
        st.stop()

    d = st.session_state.diabetes
    h = st.session_state.heart

    st.markdown("### Diabetes after cleaning")
    st.dataframe(d.head(), use_container_width=True)

    st.markdown("### Diabetes class balance before SMOTE")
    st.write(d[DIABETES_TARGET].value_counts().sort_index())

    st.markdown("### Heart Disease after cleaning")
    st.dataframe(h.head(), use_container_width=True)

    st.markdown("### Methodology configuration")
    st.write("• Physiologically implausible zeros → missing values for selected diabetes variables")
    st.write("• Class-stratified median imputation for diabetes missing values")
    st.write("• Stratified 80:20 train/test split")
    st.write("• StandardScaler")
    st.write("• SMOTE applied only inside the training pipeline")
    st.write("• Five-fold StratifiedKFold cross-validation")
    st.write("• GridSearchCV using the parameter ranges specified in Chapter Three")

elif page == "Model Training":
    st.title("Model Training")

    if st.session_state.diabetes is None or st.session_state.heart is None:
        st.warning("Upload both datasets first.")
        st.stop()

    use_smote_d = st.checkbox("Apply SMOTE to diabetes training data", value=True)
    use_smote_h = st.checkbox("Apply SMOTE to heart-disease training data", value=False)

    dataset_choice = st.radio(
        "Select dataset",
        ["Diabetes", "Heart Disease"],
        horizontal=True
    )

    if st.button("Run Model Training", type="primary"):
        with st.spinner("Training, five-fold cross-validation and GridSearchCV are running..."):
            start = time.time()

            if dataset_choice == "Diabetes":
                df = st.session_state.diabetes
                exp = run_full_experiment(
                    df[DIABETES_FEATURES], df[DIABETES_TARGET],
                    use_smote=use_smote_d
                )
                st.session_state.diabetes_exp = exp
            else:
                df = st.session_state.heart
                exp = run_full_experiment(
                    df[HEART_FEATURES], df[HEART_TARGET],
                    use_smote=use_smote_h
                )
                st.session_state.heart_exp = exp

            elapsed = time.time() - start

        st.success(f"Training completed in {elapsed:.1f} seconds.")

        st.markdown("### Model Performance")
        st.dataframe(exp["results"], use_container_width=True)

        # st.markdown("### Best hyperparameters")
        # for model_name, details in exp["tuning"].items():
        #     st.write(f"**{model_name}** — {details['Best Parameters']} | CV F1 = {details['Best CV F1']}")


elif page == "Model Evaluation":
    st.title("Model Evaluation")

    if st.session_state.diabetes is None or st.session_state.heart is None:
        st.warning("Upload both datasets first.")
        st.stop()

    dataset_choice = st.radio(
        "Dataset",
        ["Diabetes", "Heart Disease"],
        horizontal=True
    )

    exp = st.session_state.diabetes_exp if dataset_choice == "Diabetes" else st.session_state.heart_exp
    if exp is None:
        st.info("Run Model Training first.")
        st.stop()

    result_df = exp["results"]

    st.markdown("### Performance comparison")
    st.dataframe(result_df, use_container_width=True)

    best_name = result_df.sort_values(
        ["F1-Score (%)", "AUC-ROC"], ascending=False
    ).iloc[0]["Model"]

    st.success(f"Best model by F1-score/AUC priority: {best_name}")

    t1, t2 = st.tabs(["Confusion Matrix", "Classification Report"])

    with t1:
        selected_model = st.selectbox("Model", MODEL_ORDER, index=MODEL_ORDER.index("Random Forest"))
        cm = exp["cms"][selected_model]
        labels = (
            ["Non-Diabetic", "Diabetic"]
            if dataset_choice == "Diabetes"
            else ["No Disease", "Disease"]
        )

        st.dataframe(make_confusion_df(cm, labels), use_container_width=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title(f"Confusion Matrix — {selected_model}")
        st.pyplot(style_fig(fig))
        plt.close(fig)

    with t2:
        selected_model = st.selectbox("Classification report model", MODEL_ORDER)
        labels_text = (
            ["Non-Diabetic", "Diabetic"]
            if dataset_choice == "Diabetes"
            else ["No Disease", "Disease"]
        )
        report = classification_report(
            exp["y_test"],
            exp["predictions"][selected_model],
            target_names=labels_text,
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)



elif page == "Interactive Prediction":
    st.title("Interactive Clinical Prediction")
    st.warning(
        "Research/educational use only. This is not a medical diagnosis. "
        "Any clinical use requires appropriate local validation, ethical approval, and professional oversight."
    )

    choice = st.radio("Prediction task", ["Diabetes", "Heart Disease"], horizontal=True)
    exp = st.session_state.diabetes_exp if choice == "Diabetes" else st.session_state.heart_exp

    if exp is None:
        st.info("Train the models first using the Model Training page.")
        st.stop()

    best_name = st.selectbox(
        "Model",
        MODEL_ORDER,
        index=MODEL_ORDER.index("Random Forest")
    )
    model = exp["trained"][best_name]

    if choice == "Diabetes":
        c1, c2, c3 = st.columns(3)
        with c1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose", 1.0, 300.0, 120.0)
            bp = st.number_input("Blood Pressure", 1.0, 200.0, 70.0)
        with c2:
            skin = st.number_input("Skin Thickness", 1.0, 100.0, 20.0)
            insulin = st.number_input("Insulin", 1.0, 900.0, 80.0)
            bmi = st.number_input("BMI", 10.0, 80.0, 32.0)
        with c3:
            pedigree = st.number_input("Diabetes Pedigree Function", 0.01, 3.0, 0.47)
            age = st.number_input("Age", 18, 100, 33)

        row = pd.DataFrame([[
            pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age
        ]], columns=DIABETES_FEATURES)

    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 18, 100, 54)
            sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1], index=1)
            cp = st.number_input("Chest Pain Type (0–3)", 0, 3, 1)
            restbp = st.number_input("Resting BP", 80.0, 220.0, 132.0)
            chol = st.number_input("Cholesterol", 80.0, 700.0, 246.0)
        with c2:
            fbs = st.selectbox("FBS > 120 mg/dl (0/1)", [0, 1], index=0)
            restecg = st.number_input("RestECG (0–2)", 0, 2, 0)
            maxhr = st.number_input("Maximum Heart Rate", 50.0, 250.0, 150.0)
            exang = st.selectbox("Exercise-Induced Angina (0/1)", [0, 1], index=0)
        with c3:
            oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
            slope = st.number_input("Slope (0–2)", 0, 2, 1)
            ca = st.number_input("Major Vessels (0–4)", 0, 4, 0)
            thal = st.number_input("Thal (1–3)", 1, 3, 2)

        row = pd.DataFrame([[
            age, sex, cp, restbp, chol, fbs, restecg, maxhr, exang,
            oldpeak, slope, ca, thal
        ]], columns=HEART_FEATURES)

    if st.button("Predict", type="primary"):
        pred = int(model.predict(row)[0])
        prob = float(model.predict_proba(row)[0, 1])

        if choice == "Diabetes":
            label = "Diabetic" if pred == 1 else "Non-Diabetic"
        else:
            label = "Heart Disease Present" if pred == 1 else "No Heart Disease"

        st.metric("Prediction", label)
        st.metric("Estimated positive-class probability", f"{prob * 100:.1f}%")

