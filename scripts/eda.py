import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load dataset
df = pd.read_csv("data/sample_sales_data.csv")

# Convert LaunchDate to datetime
df['LaunchDate'] = pd.to_datetime(df['LaunchDate'])

# Extract month from launch date
df['Month'] = df['LaunchDate'].dt.month_name()

# -------------------------------
# 1. Basic Info
print("\n📊 Dataset Preview:")
print(df.head())

print("\n📈 Summary Statistics:")
print(df.describe())

# -------------------------------
# 2. Channel-wise average sales
channel_avg = df.groupby("Channel")["UnitsSold30"].mean().sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=channel_avg.index, y=channel_avg.values, palette="Set2")
plt.title("📢 Avg Units Sold by Channel")
plt.ylabel("Avg Units Sold (30 days)")
plt.xlabel("Marketing Channel")
plt.tight_layout()
plt.savefig("assets/channel_avg_sales.png")
plt.show()

# -------------------------------
# 3. Month-wise total sales
month_sales = df.groupby("Month")["UnitsSold30"].sum().sort_values()

plt.figure(figsize=(8,5))
sns.lineplot(x=month_sales.index, y=month_sales.values, marker="o")
plt.title("📆 Monthly Total Units Sold")
plt.ylabel("Units Sold")
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("assets/monthly_sales_trend.png")
plt.show()

# -------------------------------
# 4. Category performance
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Category", y="UnitsSold30", palette="pastel")
plt.title("📦 Category-wise Sales Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("assets/category_boxplot.png")
plt.show()
