
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from openai import OpenAI

st.set_page_config(layout="wide")
st.title("🚀 AI Data Analytics Platform")

# ==============================
# DATA UPLOAD
# ==============================

file = st.file_uploader("Upload CSV", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ==============================
    # AUTO DATA CLEANING
    # ==============================

    df = df.drop_duplicates()

    for col in df.columns:

        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])

        else:
            df[col] = df[col].fillna(df[col].mean())

    st.success("Dataset cleaned")

    # ==============================
    # EXECUTIVE DASHBOARD
    # ==============================

    st.subheader("📊 Executive Dashboard")

    numeric_cols = df.select_dtypes(include=np.number).columns

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Numeric Features", len(numeric_cols))

    if len(numeric_cols) > 0:

        fig = px.histogram(df, x=numeric_cols[0])
        st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # DRAG STYLE DASHBOARD
    # ==============================

    st.subheader("🧩 Build Custom Dashboard")

    chart_type = st.selectbox(
        "Select Chart",
        ["Histogram", "Scatter", "Line", "Box"]
    )

    column = st.selectbox("Select Column", df.columns)

    if chart_type == "Histogram":

        fig = px.histogram(df, x=column)

    elif chart_type == "Scatter":

        col2 = st.selectbox("Y Column", df.columns)
        fig = px.scatter(df, x=column, y=col2)

    elif chart_type == "Line":

        fig = px.line(df, y=column)

    else:

        fig = px.box(df, y=column)

    st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # AUTOMATIC ML MODEL SELECTION
    # ==============================

    st.subheader("🤖 AutoML Prediction")

    target = st.selectbox("Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {}

    if y.dtype == "object":

        le = LabelEncoder()

        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        models["RandomForest"] = RandomForestClassifier()

    else:

        models["RandomForest"] = RandomForestRegressor()
        models["LinearRegression"] = LinearRegression()
        models["DecisionTree"] = DecisionTreeRegressor()

    best_model = None
    best_score = -999

    for name, model in models.items():

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if y.dtype == "object":
            score = accuracy_score(y_test, preds)

        else:
            score = r2_score(y_test, preds)

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    st.success(f"Best Model: {best_name}")
    st.write("Score:", best_score)

    # ==============================
    # FEATURE IMPORTANCE
    # ==============================

    st.subheader("📌 Feature Importance")

    if hasattr(best_model, "feature_importances_"):

        imp = best_model.feature_importances_

        feat_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": imp
        }).sort_values("Importance", ascending=False)

        fig = px.bar(feat_df.head(10),
                     x="Importance",
                     y="Feature",
                     orientation="h")

        st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # FORECASTING
    # ==============================

    st.subheader("📈 Forecast Simulation")

    value_col = st.selectbox("Select Value Column", numeric_cols)

    df["rolling_mean"] = df[value_col].rolling(5).mean()

    fig = px.line(df, y=[value_col, "rolling_mean"])

    st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # GPT DATA ANALYSIS
    # ==============================

st.subheader("💬 GPT Data Analyst")

api_key = st.text_input("OpenAI API Key", type="password")
question = st.text_input("Ask a question about your dataset")

if api_key and question:

    try:
        client = OpenAI(api_key=api_key)

        prompt = f"""
        Dataset columns: {list(df.columns)}

        User question:
        {question}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        st.write(response.choices[0].message.content)

    except Exception as e:
        st.error("API Key invalid or connection failed.")

    # ==============================
    # AI REPORT GENERATION
    # ==============================

    st.subheader("📄 Generate Business Report")

    if st.button("Create Report"):

        report = "AI_Report.pdf"

        c = canvas.Canvas(report, pagesize=letter)

        c.drawString(100, 750, "AI Data Analysis Report")

        c.drawString(100, 720, f"Rows: {df.shape[0]}")
        c.drawString(100, 700, f"Columns: {df.shape[1]}")
        c.drawString(100, 680, f"Best Model: {best_name}")
        c.drawString(100, 660, f"Score: {best_score}")

        c.save()

        with open(report, "rb") as f:
            st.download_button("Download PDF", f, "AI_Report.pdf")
