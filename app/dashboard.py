import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Page Config ------------------
st.set_page_config(page_title="📦 Product Drop Optimizer", layout="wide")

# ------------------ Sidebar ------------------
st.sidebar.title("📦 Product Drop Optimizer")
st.sidebar.info("""
Smart dashboard to analyze product drops and predict sales performance.

Built with ❤️ using Python, Streamlit, and scikit-learn.
""")

# ------------------ Prediction Function ------------------
def predict_sales(price, category, channel, region):
    try:
        model = joblib.load("app/model.pkl")
        encoder = joblib.load("app/encoder.pkl")

        input_df = pd.DataFrame([{
            "Price": price,
            "Category": category,
            "Channel": channel,
            "Region": region
        }])

        encoded = encoder.transform(input_df[["Category", "Channel", "Region"]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

        X_input = pd.concat([input_df[["Price"]], encoded_df], axis=1)
        prediction = model.predict(X_input)[0]
        return int(prediction)
    except Exception as e:
        return f"⚠️ Error: {e}"

# ------------------ Title ------------------
st.title("📦 Product Drop Optimizer")
st.markdown("Upload your sales CSV and get deep insights on performance trends and predictions.")

# ------------------ Sample Download ------------------
default_df = pd.read_csv("data/sample_sales_data.csv")
st.download_button("⬇️ Download Sample CSV", data=default_df.to_csv(index=False),
                   file_name="sample_sales_data.csv", mime="text/csv")

# ------------------ File Upload ------------------
uploaded_file = st.file_uploader("📤 Upload your sales CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("⚠️ No file uploaded. Using sample data for demo.")
    df = default_df

# ------------------ Data Preview ------------------
st.subheader("📄 Data Preview")
st.dataframe(df.head())

# ------------------ Launch Day Recommender ------------------
st.subheader("📅 Best Day to Launch Recommendation")
df['LaunchDate'] = pd.to_datetime(df['LaunchDate'])
df['LaunchDay'] = df['LaunchDate'].dt.day_name()
day_avg = df.groupby('LaunchDay')['UnitsSold30'].mean().sort_values(ascending=False)

st.markdown(f"✅ **Recommended Day to Launch:** `{day_avg.idxmax()}` (Avg Units Sold: {int(day_avg.max())})")

fig1, ax1 = plt.subplots()
sns.barplot(x=day_avg.index, y=day_avg.values, ax=ax1, palette="coolwarm")
ax1.set_ylabel("Avg Units Sold")
ax1.set_title("📅 Avg Sales by Launch Day")
st.pyplot(fig1)

# ------------------ EDA Charts ------------------
st.subheader("📊 Exploratory Data Insights")

chart_paths = {
    "📢 Avg Sales by Channel": "assets/channel_avg_sales.png",
    "📆 Monthly Sales Trend": "assets/monthly_sales_trend.png",
    "📦 Sales by Category": "assets/category_boxplot.png"
}

for title, path in chart_paths.items():
    if os.path.exists(path):
        st.markdown(f"### {title}")
        st.image(Image.open(path))
    else:
        st.warning(f"❌ Chart not found: {path}")

# ------------------ Sales Predictor ------------------
st.subheader("🔮 Predict Sales for a New Product")

with st.form("predict_form"):
    price = st.number_input("Product Price ($)", min_value=0.0, value=49.99)
    category = st.selectbox("Product Category", ["Electronics", "Fitness", "Home Decor", "Kitchen", "Home Appliances"])
    channel = st.selectbox("Marketing Channel", ["Instagram", "Email", "Facebook", "YouTube"])
    region = st.selectbox("Region", ["US", "UK", "Canada", "India"])

    submitted = st.form_submit_button("Predict Units Sold")

    if submitted:
        result = predict_sales(price, category, channel, region)
        if isinstance(result, int):
            st.success(f"📦 Predicted Units Sold in 30 Days: **{result}**")
        else:
            st.error(result)

# ------------------ ROI Calculator ------------------
st.markdown("---")
st.subheader("💰 ROI Calculator")

with st.form("roi_form"):
    price = st.number_input("Unit Price ($)", min_value=0.0, value=49.99, key="roi_price")
    ad_spend = st.number_input("Marketing Spend ($)", min_value=1.0, value=100.0, key="roi_spend")
    category = st.selectbox("Product Category", ["Electronics", "Fitness", "Home Decor", "Kitchen", "Home Appliances"], key="roi_category")
    region = st.selectbox("Region", ["US", "UK", "Canada", "India"], key="roi_region")

    roi_submit = st.form_submit_button("Calculate ROI")

    if roi_submit:
        channels = ["Instagram", "Email", "Facebook", "YouTube"]
        roi_results, units_sold_list = [], []

        for ch in channels:
            predicted_units = predict_sales(price, category, ch, region)
            if isinstance(predicted_units, int):
                revenue = predicted_units * price
                roi = ((revenue - ad_spend) / ad_spend) * 100
                roi_results.append(roi)
                units_sold_list.append(predicted_units)
            else:
                roi_results.append(0)
                units_sold_list.append(0)

        best_idx = np.argmax(roi_results)
        best_channel = channels[best_idx]
        best_roi = roi_results[best_idx]

        st.success(f"📈 Best ROI Channel: **{best_channel}** with ROI of **{best_roi:.2f}%**")

        fig, ax = plt.subplots()
        ax.bar(channels, roi_results, color="skyblue")
        ax.set_title("Channel-wise ROI Comparison")
        ax.set_ylabel("ROI (%)")
        ax.axhline(0, color='gray', linestyle='--')
        st.pyplot(fig)

        st.markdown("### 📋 ROI Breakdown Table")
        roi_table = pd.DataFrame({
            "Channel": channels,
            "Predicted Units Sold": units_sold_list,
            "ROI (%)": [round(r, 2) for r in roi_results]
        })
        st.dataframe(roi_table)

# ------------------ Batch Prediction ------------------
st.markdown("---")
st.subheader("📤 Batch Prediction: Upcoming Products")

st.markdown("Upload a CSV with upcoming product launches to get predicted sales.")

sample_batch = pd.DataFrame({
    "Price": [49.99, 29.99],
    "Category": ["Electronics", "Fitness"],
    "Channel": ["Instagram", "Email"],
    "Region": ["US", "Canada"]
})

st.download_button("⬇️ Download Batch Sample CSV", sample_batch.to_csv(index=False), file_name="batch_input_sample.csv")

batch_file = st.file_uploader("Upload Batch CSV", type=["csv"], key="batch")

if batch_file:
    batch_df = pd.read_csv(batch_file)
    predictions, rois = [], []

    for i, row in batch_df.iterrows():
        pred = predict_sales(row["Price"], row["Category"], row["Channel"], row["Region"])
        predictions.append(pred)

        if "AdSpend" in batch_df.columns:
            roi = ((pred * row["Price"] - row["AdSpend"]) / row["AdSpend"]) * 100
            rois.append(round(roi, 2))

    batch_df["Predicted_Units_Sold"] = predictions
    if rois:
        batch_df["ROI (%)"] = rois

    st.success("✅ Batch Predictions Done")
    st.dataframe(batch_df)

    csv = batch_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results CSV", data=csv, file_name="batch_predictions.csv", mime="text/csv")

# ------------------ How It Works ------------------
st.markdown("---")
st.markdown("## 📘 How it Works")
st.markdown("""
1. Upload your past product sales CSV  
2. Explore launch day & category insights  
3. Predict units sold for new products  
4. Find most profitable channel using ROI tool  
5. Upload batch of upcoming products for bulk predictions  
""")
