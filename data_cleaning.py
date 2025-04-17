#%%
# This file is only for creating cleaned_churn_data.csv
import pandas as pd

df = pd.read_csv("Dataset/telecom_customer_churn.csv")
# Removes "Joined" customers
df_no_j = df[df["Customer Status"] != "Joined"]

# %%
# Adds average monthly revenue column
df_no_j.columns.get_loc("Total Revenue")
avg_mo_rev = df_no_j["Total Revenue"] / df_no_j["Tenure in Months"]
df_no_j.insert(loc=35, column="Avg Monthly Revenue", value=avg_mo_rev)
# %%
df_no_j["Tenure in Months"].value_counts()
# %%
import matplotlib.pyplot as plt
plt.hist(df_no_j["Avg Monthly Revenue"])
plt.hist(df_no_j["Age"])
# %%
df_no_j.to_csv("Dataset/cleaned_churn_data.csv", index=False)
# %%
df_no_j["Avg Monthly Revenue"] = df_no_j["Avg Monthly Revenue"].round(2)
# %%
# Writes to csv
df_no_j.to_csv("Dataset/cleaned_churn_data.csv", index=False)
# %%
