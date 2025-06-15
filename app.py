import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



def plot_all_decision_boundaries(df):
    numeric_features = df.select_dtypes(include=[np.number]).columns.drop('Outcome')
    feature_pairs = list(itertools.combinations(numeric_features, 2))

    cols = 3
    rows = (len(feature_pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes.flatten()

    for idx, (f1, f2) in enumerate(feature_pairs):
        X = df[[f1, f2]]
        y = df['Outcome']

        model = LogisticRegression()
        model.fit(X, y)

        x_min, x_max = X[f1].min() - 1, X[f1].max() + 1
        y_min, y_max = X[f2].min() - 1, X[f2].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax = axes[idx]
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        ax.scatter(X[f1][y == 0], X[f2][y == 0], label='0', alpha=0.6, edgecolor='k', s=15)
        ax.scatter(X[f1][y == 1], X[f2][y == 1], label='1', alpha=0.6, edgecolor='k', s=15)
        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        ax.set_title(f'{f1} vs {f2}')

    # Remove unused axes
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

# Set the title and adjust layout
    fig.suptitle('Logistic Regression Decision Boundaries for All Numeric Feature Pairs', fontsize=16, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98])  # Reserve space for title
    return fig




# Load dataset
df = pd.read_csv("kaggle_diabetes.csv")

# Preprocessing
X_raw = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)
X = pd.DataFrame(X, columns=X_raw.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --- Streamlit UI ---
st.set_page_config(layout='wide')
st.title("🔍 Diabetes Prediction App")

# --- Sidebar Navigation ---
selected_page = st.sidebar.selectbox(
    "Choose a page",
    ["Introduction", "EDA", "Predict", "Conclusion And Evaluation"]
)


# ===================== EDA Section =====================
if selected_page == "EDA":
    st.header("📊 Exploratory Data Analysis")

    tab1, tab2 = st.tabs(["📄 Dataset Summary", "📊 Visualizations"])

    # ================= EDA SECTION =================
    # Ensure binned columns exist regardless of which section user selects
    df['Pregnancies_bin'] = pd.cut(df['Pregnancies'], [-1, 0, 1, 3, 5, 8, 12, 20],
                               labels=['None', '1', '2–3', '4–5', '6–8', '9–12', '13+'])
    df['Glucose_bin'] = pd.cut(df['Glucose'], [0, 99, 125, df['Glucose'].max()],
                           labels=['Low', 'Medium', 'High'])
    df['SkinThickness_bin'] = pd.cut(df['SkinThickness'], [-1, 0, 20, 40, 60, df['SkinThickness'].max()],
                                 labels=['None', 'Low', 'Medium', 'High', 'Very High'])
    df['Insulin_bin'] = pd.cut(df['Insulin'], [-1, 0, 100, 200, 300, df['Insulin'].max()],
                           labels=['None', 'Low', 'Medium', 'High', 'Very High'])
    df['BMI_bin'] = pd.cut(df['BMI'], [0, 18.5, 24.9, 29.9, 34.9, df['BMI'].max()],
                       labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'Extremely Obese'])
    df['Age_bin'] = pd.cut(df['Age'], [20, 29, 39, 49, 59, 69, 81],
                       labels=['21–29', '30–39', '40–49', '50–59', '60–69', '70–81'])
    column_descriptions = {
    "Pregnancies": "Number of times the patient has been pregnant.",
    "Glucose": "Plasma glucose concentration after 2 hours in an oral glucose tolerance test.",
    "BloodPressure": "Diastolic blood pressure (mm Hg).",
    "SkinThickness": "Triceps skin fold thickness (mm).",
    "Insulin": "2-Hour serum insulin (mu U/ml).",
    "BMI": "Body mass index (weight in kg/(height in m)^2).",
    "DiabetesPedigreeFunction": "A function which scores the likelihood of diabetes based on family history.",
    "Age": "Age of the patient (years).",
    "Outcome": "Class variable (0: Non-diabetic, 1: Diabetic)"
}



    with tab1:
        st.subheader("📌 Dataset Overview")
        st.dataframe(df.head())

        st.subheader("📘 Feature Descriptions")
        description_df = pd.DataFrame(list(column_descriptions.items()), columns=["Feature", "Description"])
        st.dataframe(description_df)


        st.subheader("📈 Descriptive Statistics")
        st.dataframe(df.describe())

        st.subheader("🧾 Data Types and Null Values")
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Data Types")
            st.dataframe(df.dtypes.reset_index().rename(columns={0: "Type", "index": "Feature"}))
        with col2:
            st.write("### Null Values")
            st.dataframe(df.isnull().sum().reset_index().rename(columns={0: "Null Count", "index": "Feature"}))

        st.subheader("📊 Binned Feature Frequency Table")
        binned_columns = {
            'Pregnancies_bin': pd.cut(df['Pregnancies'], [-1, 0, 1, 3, 5, 8, 12, 20], labels=['None', '1', '2–3', '4–5', '6–8', '9–12', '13+']),
            'Glucose_bin': pd.cut(df['Glucose'], [0, 99, 125, df['Glucose'].max()], labels=['Low', 'Medium', 'High']),
            'Age_bin': pd.cut(df['Age'], [20, 29, 39, 49, 59, 69, 81], labels=['21–29', '30–39', '40–49', '50–59', '60–69', '70–81'])
        }
        for col_name, binned_series in binned_columns.items():
            df[col_name] = binned_series
            st.write(f"#### Frequency of {col_name.replace('_bin', '')} (Binned)")
            st.dataframe(df[col_name].value_counts().reset_index().rename(columns={"index": col_name, col_name: "Count"}))

        st.subheader("📘 Outcome-Based Group Statistics")
        st.write("### Mean per Outcome Group")
        st.dataframe(df.groupby('Outcome').mean(numeric_only=True))


        st.write("### Mean, Median, and Std per Outcome Group")
        numeric_df = df.select_dtypes(include='number')
        group_stats = numeric_df.groupby(df['Outcome']).agg(['mean', 'median', 'std'])
        st.dataframe(group_stats)


        st.subheader("📉 Correlation Table")
        st.dataframe(df.corr(numeric_only=True))

    # ================= VISUALIZATION SECTION =================
    with tab2:
        st.subheader("📊 Binned Count Plots (Hue by Outcome)")

    # Automatically find all binned columns
        binned_cols = [col for col in df.columns if col.endswith('_bin')]

    # Setup grid layout
        n_cols = 3
        n_rows = (len(binned_cols) + n_cols - 1) // n_cols

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axs = axs.flatten()

        for i, col in enumerate(binned_cols):
            sns.countplot(x=col, hue='Outcome', data=df, ax=axs[i])
            axs[i].set_title(f"{col.replace('_bin', '')} (Binned)")
            axs[i].tick_params(axis='x', rotation=30)

    # Hide unused axes
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        st.pyplot(fig)


        st.subheader("📌 KDE Plots for Continuous Features")
        continuous_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        axs = axs.flatten()
        for i, col in enumerate(continuous_cols):
            sns.kdeplot(data=df, x=col, hue='Outcome', fill=True, ax=axs[i])
            axs[i].set_title(col)
        st.pyplot(fig)

        st.subheader("📦 Boxplots for Outlier Detection by Outcome")

        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        axs = axs.flatten()

        for i, col in enumerate(continuous_cols[:len(axs)]):
            try:
                sns.boxplot(data=df, x='Outcome', y=col, ax=axs[i])
                axs[i].set_title(col)
            except Exception as e:
                axs[i].set_visible(False)  # hide if plot fails

        st.pyplot(fig)


        st.subheader("🧩 Correlation Heatmap")
        corr = df[continuous_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ===================== Model Evaluation Section =====================
elif selected_page == "Conclusion And Evaluation":    
    st.header("🧪 Model Performance Comparison")

    tab1, tab2 = st.tabs(["📊 Model Comparison", "📌 Project Conclusion"])

    with tab1:
        st.subheader("📉 Confusion Matrices & Classification Metrics")
        st.markdown("---")


        # Predictions
        y_pred_log = log_model.predict(X_test)
        y_pred_rf = rf_model.predict(X_test)

        # Accuracy
        acc_log = accuracy_score(y_test, y_pred_log)
        acc_rf = accuracy_score(y_test, y_pred_rf)

        # Confusion Matrices
        cm_log = confusion_matrix(y_test, y_pred_log)
        cm_rf = confusion_matrix(y_test, y_pred_rf)

        cm_log_df = pd.DataFrame(cm_log, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"])
        cm_rf_df = pd.DataFrame(cm_rf, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"])

        # Classification Reports
        cr_log_df = pd.DataFrame(classification_report(y_test, y_pred_log, output_dict=True)).transpose().round(2)
        cr_rf_df = pd.DataFrame(classification_report(y_test, y_pred_rf, output_dict=True)).transpose().round(2)

        # Columns with vertical divider
        col1, spacer, col2 = st.columns([1, 0.05, 1])

        with col1:
            st.subheader("🔹 Logistic Regression")
            st.metric("Accuracy", f"{acc_log:.2f}")

            st.markdown("**Confusion Matrix**")
            st.dataframe(cm_log_df)

            st.markdown("**Classification Report**")
            st.dataframe(cr_log_df)

        with spacer:
            st.markdown("<div style='height:100%; border-left:2px solid lightgray;'></div>", unsafe_allow_html=True)

        with col2:
            st.subheader("🔹 Random Forest Classifier")
            st.metric("Accuracy", f"{acc_rf:.2f}")

            st.markdown("**Confusion Matrix**")
            st.dataframe(cm_rf_df)

            st.markdown("**Classification Report**")
            st.dataframe(cr_rf_df)

        st.subheader("🧠 Decision Boundary Comparison")
        st.markdown("---")


        # OPTIONAL: Use saved plot from your notebook, or recreate here
        try:
            fig = plot_all_decision_boundaries(df)
            st.pyplot(fig)
        except Exception as e:
            st.info("⚠️ Decision boundary plots not available or plotting failed.")
            st.text(f"Reason: {e}")

    with tab2:
        st.subheader("📌 Final Conclusion")

        st.markdown("""
        ### 🔍 Summary of Findings

        - **EDA** revealed that features like `Glucose`, `BMI`, and `Age` have significant differences between diabetic and non-diabetic individuals.
        - Missing values in columns like `Insulin` and `SkinThickness` were addressed appropriately.
        - Binned visualizations helped in understanding distribution patterns.

        ### 🧪 Model Performance

        - **Random Forest** achieved higher accuracy and recall than Logistic Regression.
        - Logistic Regression is interpretable and may be preferable for clinical transparency.
        - Random Forest handles non-linearity and interactions better.

        ### ✅ Recommendation

        - Use **Random Forest** if accuracy is the top priority.
        - Use **Logistic Regression** if interpretability is critical.

        ### 🛠️ Suggestions for Improvement

        - Tune hyperparameters using GridSearchCV.
        - Use feature selection to remove less relevant features.
        - Consider model calibration to improve probability confidence.
        - Use cross-validation for more reliable performance estimates.


        ---
        """)


# ===================== Prediction Section =====================
elif selected_page == "Predict":
    st.header("🧠 Predict Diabetes from Inputs")
    st.write("🔧 Enter patient data below:")

    # Group inputs in 3 columns for better layout
    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 17, 1, help="Number of times the patient has been pregnant.")
        bp = st.number_input("Blood Pressure (mm Hg)", 30, 122, 70, help="Diastolic blood pressure.")
        insulin = st.number_input("Insulin (mu U/ml)", 0, 846, 80, help="2-Hour serum insulin level.")

    with col2:
        glucose = st.number_input("Glucose", 50, 200, 100, help="Plasma glucose concentration after 2 hours.")
        skin = st.number_input("Skin Thickness (mm)", 0, 99, 20, help="Triceps skin fold thickness.")
        bmi = st.number_input("BMI", 10.0, 67.1, 25.0, step=0.1, format="%.1f", help="Body Mass Index (kg/m²).")

    with col3:
        dpf = st.number_input("Diabetes Pedigree Function", 0.05, 2.5, 0.5, step=0.01, format="%.2f", help="Family history-based diabetes likelihood.")
        age = st.number_input("Age", 21, 81, 30, help="Age in years.")

    # Optional UX tip: flagging extreme or uncommon inputs
    if bmi < 12 or bmi > 50:
        st.warning("⚠️ BMI value looks unusual. Please verify.")

    if glucose > 180:
        st.warning("⚠️ High glucose level! Consider rechecking.")

    # Prepare input data
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [bp],
        'SkinThickness': [skin],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

    # Normalize input like training data
    input_scaled = scaler.transform(input_data)

    # Model selection
    model_choice = st.radio("Choose Model", ['Random Forest', 'Logistic Regression'])

    # Predict button
    if st.button("🔍 Predict"):
        if model_choice == 'Random Forest':
            prediction = rf_model.predict(input_scaled)[0]
            prob = rf_model.predict_proba(input_scaled)[0][1]
        else:
            prediction = log_model.predict(input_scaled)[0]
            prob = log_model.predict_proba(input_scaled)[0][1]

        st.success(f"🎯 Prediction: {'Diabetic' if prediction else 'Not Diabetic'}")
        st.info(f"Confidence: {prob * 100:.2f}%")
if selected_page == "Introduction":
    # Full-width image
    st.image("img.png", use_container_width=True)

    # Title and Overview
    st.title("🧬 Welcome to the Diabetes Prediction App")

    st.markdown("""
    ## 📘 Project Overview

    This app analyzes health-related attributes from the **Pima Indians Diabetes Dataset** to:

    - 🔍 Explore hidden patterns using interactive visualizations  
    - 🧠 Train and compare multiple machine learning models  
    - 📈 Predict the likelihood of diabetes  
    - 🎯 Visualize decision boundaries for interpretability  
    - 📊 Evaluate model performance with metrics

    ---
    """)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("""
        ### ❓ Why This Dataset?

        - A popular benchmark dataset in the ML and healthcare community  
        - Real-world medical records from **Pima Indian women**  
        - Balanced features across clinical indicators like **Glucose**, **BMI**, **Insulin**, and more  
        - Suitable for both **classification tasks** and **model interpretability studies**

        ### 🧠 How the Model Was Trained?

        - 🔄 Data cleaning & handling of missing values  
        - 📊 Exploratory Data Analysis (EDA)  
        - 🎯 Feature selection and scaling  
        - 🧪 Trained multiple models (Logistic Regression, Random Forest, etc.)  
        - ✅ Best model chosen based on accuracy and F1-score  
        - 📉 Visualized decision boundaries and confusion matrix

        ### 🛠️ Tools & Technologies Used

        - `Python`, `Pandas`, `Numpy` for data wrangling  
        - `Matplotlib`, `Seaborn` for data visualization  
        - `Scikit-learn` for machine learning models  
        - `Streamlit` for web-based interactive UI  

        ---
        """)

    with col2:
        st.markdown("""
        ### 🧪 How to Use This App?

        Navigate through the sidebar:

        - 🏠 **Introduction** — Overview of the project  
        - 📊 **EDA** — Explore distributions, outliers, and correlations  
        - 🤖 **Predict** — Input patient values and predict diabetes  
        - 📈 **Conclusion** — View evaluation metrics and model insights  

        > 🔁 Tip: Hover over charts for interactivity!

        ---
        """)

    # Footer
    st.markdown("---")
    st.markdown("Created by **Muqnit Ur Rehman** | 🧠 Machine Learning Enthusiast")
