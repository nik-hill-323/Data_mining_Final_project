#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
tcc = pd.read_csv("telecom_customer_churn.csv")
# %%
print(tcc["Phone Service"].unique())
print(tcc["Internet Service"].unique())
# %%
tcc["Churn Category"].unique()
# %%
tcc["Churn Reason"].unique()
# %%
plt.hist(tcc["Total Revenue"])
# %%
has_phone = tcc["Phone Service"] == "Yes"
has_internet = tcc["Internet Service"] == "Yes"

both = sum(has_phone & has_internet)
phone_only = sum(has_phone & ~has_internet)
internet_only = sum(~has_phone & has_internet)
neither = sum(~has_phone & ~has_internet)

print(both + phone_only + internet_only)
plt.pie([both, phone_only, internet_only], autopct='%1.1f%%', 
        labels=["both", "phone only", "Internet only"])
plt.title("Services Customers Are Subscribed To")
plt.show()
# %%
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Step 1: Your list of ZIP codes (some repeated)
zip_list = tcc['Zip Code'].astype(str).tolist()

# Step 2: Count how many times each ZIP appears
zip_counts = pd.DataFrame(Counter(zip_list).items(), columns=['ZIP', 'count'])

# Step 3: Load ZIP code shapefile (ZCTA)
shapefile_path = 'tl_2022_us_zcta520.shp'  # Update with your file path
zcta = gpd.read_file(shapefile_path)
zcta['ZIP'] = zcta['ZCTA5CE20']  # Rename for consistency

# ✅ Step 4: Filter to ZIP codes starting with '9' only
zcta = zcta[zcta['ZIP'].str.startswith(('90', '91', '92', '93', '94', '95', '960', '961'))]

# Step 5: Merge counts with ZIP shapes
zcta = zcta.merge(zip_counts, on='ZIP', how='left')
zcta['count'] = zcta['count'].fillna(0)

# Step 6: Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
zcta.plot(column='count', ax=ax, legend=True, cmap='Blues', edgecolor='gray', linewidth=0.2)
ax.set_title('Frequency of ZIP Codes Starting with 9')
ax.axis('off')
plt.tight_layout()
plt.show()


# %%
