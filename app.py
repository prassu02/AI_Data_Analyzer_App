
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

st.set_page_config(layout="wide")
st.title("🚀 AI Data Analytics Platform")

# =============================
# FILE UPLOAD
# =============================

uploaded_file = st.file_uploader(
    "Upload CSV or Excel Dataset",
    type=["csv","xlsx"]
)

if uploaded_file:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset loaded")

    file_size = uploaded_file.size/(1024*1024)
    st.info(f"File Size: {file_size:.2f} MB")

    # =============================
    # AUTO DATA CLEANING
    # =============================

    df = df.drop_duplicates()

    for col in df.columns:

        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])

        else:
            df[col] = df[col].fillna(df[col].mean())

    # =============================
    # DATA PREVIEW
    # =============================

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # =============================
    # AUTO DATA INSIGHTS
    # =============================

    st.subheader("Automatic Insights")

    col1,col2,col3 = st.columns(3)

    col1.metric("Rows",df.shape[0])
    col2.metric("Columns",df.shape[1])
    col3.metric("Missing Values",df.isnull().sum().sum())

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols)>0:

        fig = px.histogram(df,x=numeric_cols[0])
        st.plotly_chart(fig,use_container_width=True)

    # =============================
    # DRAG STYLE DASHBOARD
    # =============================

    st.subheader("Build Dashboard")

    chart = st.selectbox(
        "Chart Type",
        ["Histogram","Scatter","Line","Box"]
    )

    x = st.selectbox("X Column",df.columns)

    if chart=="Histogram":

        fig = px.histogram(df,x=x)

    elif chart=="Scatter":

        y = st.selectbox("Y Column",df.columns)
        fig = px.scatter(df,x=x,y=y)

    elif chart=="Line":

        fig = px.line(df,y=x)

    else:

        fig = px.box(df,y=x)

    st.plotly_chart(fig,use_container_width=True)

    # =============================
    # AUTOML MODEL TRAINING
    # =============================

    st.subheader("AutoML Prediction")

    target = st.selectbox("Target Column",df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X)

    if y.dtype=="object":

        le = LabelEncoder()
        y = le.fit_transform(y)

        model = RandomForestClassifier()

        param = {
            "n_estimators":[50,100],
            "max_depth":[3,5,10]
        }

        metric = accuracy_score

    else:

        model = RandomForestRegressor()

        param = {
            "n_estimators":[50,100],
            "max_depth":[3,5,10]
        }

        metric = r2_score

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    grid = GridSearchCV(model,param,cv=3)

    grid.fit(X_train,y_train)

    best_model = grid.best_estimator_

    pred = best_model.predict(X_test)

    score = metric(y_test,pred)

    st.success(f"Best Model Score: {score}")

    # =============================
    # FEATURE IMPORTANCE
    # =============================

    if hasattr(best_model,"feature_importances_"):

        st.subheader("Feature Importance")

        imp = best_model.feature_importances_

        feat = pd.DataFrame({
            "Feature":X.columns,
            "Importance":imp
        }).sort_values("Importance",ascending=False)

        fig = px.bar(feat.head(10),
                     x="Importance",
                     y="Feature",
                     orientation="h")

        st.plotly_chart(fig,use_container_width=True)

    # =============================
    # FORECAST SIMULATION
    # =============================

    st.subheader("Forecast Simulation")

    if len(numeric_cols)>0:

        value_col = st.selectbox(
            "Select Value Column",
            numeric_cols
        )

        df["rolling_mean"] = df[value_col].rolling(5).mean()

        fig = px.line(df,y=[value_col,"rolling_mean"])

        st.plotly_chart(fig,use_container_width=True)

    # =============================
    # DATASET CHAT (NO API)
    # =============================

    st.subheader("Chat with Dataset")

    question = st.text_input("Ask a question")

    if question:

        q = question.lower()

        if "average" in q:

            st.write(df.mean(numeric_only=True))

        elif "max" in q:

            st.write(df.max(numeric_only=True))

        elif "min" in q:

            st.write(df.min(numeric_only=True))

        elif "correlation" in q:

            st.write(df.corr(numeric_only=True))

        else:

            st.write("Try asking about average, max, min, correlation.")

    # =============================
    # BUSINESS REPORT
    # =============================

    st.subheader("Generate Business Report")

    if st.button("Create PDF Report"):

        report="AI_Report.pdf"

        c = canvas.Canvas(report,pagesize=letter)

        c.drawString(100,750,"AI Data Analysis Report")
        c.drawString(100,720,f"Rows: {df.shape[0]}")
        c.drawString(100,700,f"Columns: {df.shape[1]}")
        c.drawString(100,680,f"Model Score: {score}")

        c.save()

        with open(report,"rb") as f:

            st.download_button(
                "Download Report",
                f,
                file_name="AI_Report.pdf"
            )

else:

    st.info("Upload a dataset to begin analysis.")
    
