
import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend (container name "backend" on the shared Docker network)
BACKEND_URL = "http://backend:7860"

st.set_page_config(page_title="SuperKart Sales Prediction", layout="centered")
st.title("SuperKart System")
st.write("Enter the product and store details below to predict the total sales of a product in a store.")

# ---- Sidebar: backend status (proves which model the UI is talking to) ----
with st.sidebar:
    st.header("Backend status")
    try:
        info = requests.get(f"{BACKEND_URL}/v1/info", timeout=5).json()
        st.success("Backend online")
        st.write(f"**Model:** {info.get('model')}")
        st.write(f"**Model MD5:** `{str(info.get('model_md5'))[:12]}…`")
        st.write(f"**scikit-learn:** {info.get('scikit-learn')}  \n**xgboost:** {info.get('xgboost')}  \n**Python:** {info.get('python')}")
    except Exception as e:
        st.error(f"Backend not reachable: {e}")

# ---- Single prediction ----
st.subheader("Single Prediction")
col1, col2 = st.columns(2)
with col1:
    Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
    Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.027, format="%.3f")
    Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
    Product_Id_char = st.selectbox("Product ID Character", ["FD", "DR", "NC"], help="FD = Food, DR = Drinks, NC = Non-Consumable")
with col2:
    Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"], index=1)
    Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"], index=1)
    Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"], index=1)
    Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, value=16)
    Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"], index=1)

product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category,
}

if st.button("Predict", type="primary"):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=product_data, timeout=30)
        if response.status_code == 200:
            predicted_sales = response.json()["Sales"]
            st.success(f"Predicted Product Store Sales Total: {predicted_sales:,.2f}")
            st.json(product_data, expanded=False)
        else:
            st.error(f"API error {response.status_code}: {response.json().get('error', response.text)}")
    except Exception as e:
        st.error(f"Unable to connect to the prediction API: {e}")

# ---- Batch prediction ----
st.subheader("Batch Prediction")
st.caption("Upload a CSV with the 10 feature columns (same format as Batch_Data_SuperKart.csv).")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(input_df)} rows")
    st.dataframe(input_df.head(), use_container_width=True)

    if st.button("Predict for Batch", type="primary"):
        try:
            uploaded_file.seek(0)
            response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files={"file": uploaded_file}, timeout=60)
            if response.status_code == 200:
                results = response.json()
                preds = pd.Series(results).astype(float)
                preds.index = preds.index.astype(int)
                result_df = input_df.copy()
                result_df["Predicted_Sales"] = preds.sort_index().values
                st.success(f"Predictions completed for {len(result_df)} rows")
                st.dataframe(result_df, use_container_width=True)
                st.download_button("Download predictions as CSV",
                                   result_df.to_csv(index=False).encode("utf-8"),
                                   "superkart_predictions.csv", "text/csv")
            else:
                st.error(f"API error {response.status_code}: {response.json().get('error', response.text)}")
        except Exception as e:
            st.error(f"Unable to connect to the prediction API: {e}")
