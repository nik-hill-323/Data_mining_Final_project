#%%
"""
This file makes maps of customer locations.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
"""
This portion uses the original dataset (including joined customers).
"""
tcc = pd.read_csv("Dataset/telecom_customer_churn.csv")
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
shapefile_path = 'Dataset/GeographicFiles/tl_2022_us_zcta520.shp' 
zcta = gpd.read_file(shapefile_path)
zcta['ZIP'] = zcta['ZCTA5CE20'] 

# ✅ Step 4: Filter to ZIP codes starting with '9' only
zcta = zcta[zcta['ZIP'].str.startswith(('90', '91', '92', '93', '94', '95', '960', '961'))]

# Step 5: Merge counts with ZIP shapes
zcta = zcta.merge(zip_counts, on='ZIP', how='left')
zcta['count'] = zcta['count'].fillna(0)
#%%
# Step 6: Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
zcta.plot(column='count', ax=ax, legend=True, cmap='Blues', edgecolor='gray', linewidth=0.1)
ax.set_title('Frequency of ZIP Codes Starting with 9')
ax.axis('off')
plt.tight_layout()
plt.show()


# %%
# HEATMAP (IGNORE)
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import seaborn as sns

# Example lat/lon list
coords = list(zip(tcc['']))

# Convert to GeoDataFrame
df = pd.DataFrame(coords, columns=['lat', 'lon'])
geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

# Project to a metric CRS (important for KDE)
gdf = gdf.to_crs(epsg=3310)  # California Albers

# Plot KDE heatmap using seaborn
x = gdf.geometry.x
y = gdf.geometry.y

plt.figure(figsize=(8, 8))
sns.kdeplot(x=x, y=y, cmap="Reds", shade=True, bw_adjust=0.5)
plt.title("California Heatmap of Points")
plt.axis("off")
plt.tight_layout()
plt.show()

# %%
import geopandas as gpd
from shapely.geometry import Point

counties = gpd.read_file("Dataset/GeographicFiles/tl_2024_us_county.shp")
ca_counties = counties[counties['STATEFP'] == '06']

geometry = [Point(xy) for xy in zip(tcc['Longitude'], tcc['Latitude'])]
gdf_points = gpd.GeoDataFrame(tcc, geometry=geometry, crs='EPSG:4326')  # WGS 84

ca_counties = ca_counties.to_crs(gdf_points.crs)

points_with_county = gpd.sjoin(gdf_points, ca_counties, how='inner', predicate='within')

point_counts = points_with_county['NAME'].value_counts().reset_index()
point_counts.columns = ['County', 'Count']

ca_counties = ca_counties.merge(point_counts, how='left', left_on='NAME', right_on='County')
ca_counties['Count'] = ca_counties['Count'].fillna(0)  # counties with 0 points

fig, ax = plt.subplots(figsize=(10, 12))
ca_counties.plot(column='Count', ax=ax, legend=True, cmap='OrRd', edgecolor='black')
ax.set_title('Number of Data Points per County in California', fontsize=14)
ax.axis('off')
plt.show()
# %%
type(ca_counties)
# %%
revenue_table = df.pivot_table(
    values='Avg Monthly Revenue',
    index='Age Bracket',
    columns='Gender',
    aggfunc='mean'
).round(2)

# Step 2: Plot
ax = revenue_table.plot(kind='bar', figsize=(10, 6), colormap='tab20c', edgecolor='black')

plt.title('Avg Monthly Revenue by Age Bracket and Gender')
plt.ylabel('Average Monthly Revenue ($)')
plt.xlabel('Age Bracket')
plt.xticks(rotation=45)
plt.legend(title='Gender')
plt.tight_layout()
plt.show()

# %%
