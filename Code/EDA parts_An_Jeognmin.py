#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
# Reading the csv
data = pd.read_csv("telecom_customer_churn.csv")
print(data.head())

#%%
# Check the data type and missing values ​​of the column
print(data['Customer Status'].unique())
print(data['Customer Status'].isnull().sum())
print(data['Customer Status'].dtype)

#%%
## Histograms
# Age Distribution
plt.figure(figsize=(6, 4))
sns.histplot(data, x='Age', kde=True, bins=30, hue='Customer Status')
plt.title('Distribution of Customer Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Tenure in Months
plt.figure(figsize=(6, 4))
sns.histplot(data, x='Tenure in Months', kde=True, bins=30, hue='Customer Status')
plt.title('Distribution of Tenure (in Months)')
plt.xlabel('Tenure in Months')
plt.ylabel('Count')
plt.show()

# Montly Charge
plt.figure(figsize=(6, 4))
sns.histplot(data, x='Monthly Charge', kde=True, bins=30, hue='Customer Status')
plt.title('Monthly Charge by Customer Status')
plt.xlabel('Monthly Charge ($)')
plt.ylabel('Count')
plt.show()

# Avg Montly GB Download
plt.figure(figsize=(6, 4))
sns.histplot(data, x='Avg Monthly GB Download', kde=True, bins=30, hue='Customer Status')
plt.title('Avg Monthly GB Download by Customer Status')
plt.xlabel('Avg Monthly GB Download (GB)')
plt.ylabel('Count')
plt.show()

# Total Revenue
plt.figure(figsize=(6, 4))
sns.histplot(data, x='Total Revenue', kde=True, bins=30, hue='Customer Status')
plt.title('Distribution of Total Revenue')  
plt.xlabel('Total Revenue ($)')
plt.ylabel('Count')
plt.show()


## joined?????
# Tenure in Months: Churned customers tend to have shorter subscription periods - dissatisfaction occurs early in the customer journey.
# Monthly Charge: Churned customers tend to have relatively more high-cost customers - may reflect unmet expectations for premium service.
# Avg Monthly GB Download: Churn can also show a pattern of decreasing for customers with higher usage volumes -> but, those customers are often more sensitive to service disruptions.
# Total Revenue: Churned customers are mostly found in the lower revenue range ->  preventing churn among high-revenue customers remains still a priority.


#%%
## Histograms using subplots (we have to choose.)
# List of variables to draw histograms
hist_columns = ['Age', 'Tenure in Months', 'Monthly Charge', 'Total Revenue', 'Avg Monthly GB Download']
# Draw histograms
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

for idx, col in enumerate(hist_columns):
    ax = axes[idx // 3, idx % 3]
    sns.histplot(data, x=col, hue='Customer Status', kde=True, bins=30, ax=ax)
    ax.set_title(f'{col} by Customer Status')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')

# Remove empty subplot
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()






# %%
## Pie Charts
# Gender Distribution
plt.figure(figsize=(6, 6))
gender_counts = data['Gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Gender Distribution')
plt.show()

# Contract Type Distribution
plt.figure(figsize=(6, 6))
contract_counts = data['Contract'].value_counts()
plt.pie(contract_counts, labels=contract_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Contract Type Distribution')
plt.show()

# Payment Method Distribution
plt.figure(figsize=(6, 6))
payment_counts = data['Payment Method'].value_counts()
plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Payment Method Distribution')
plt.show()

# Internet Type Distribution
plt.figure(figsize=(6, 6))
internet_counts = data['Internet Type'].value_counts()
plt.pie(internet_counts, labels=internet_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Internet Type Distribution')
plt.show()

# Offer 
plt.figure(figsize=(6, 6))
offer_counts = data['Offer'].value_counts()
plt.pie(offer_counts, labels=offer_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Offer Distribution')
plt.show()

#%%
## Pie charts using subplots (we have to choose.)

# List of categorical variables to draw in pie chart
pie_columns = ['Gender', 'Contract', 'Payment Method', 'Internet Type']
# Draw pie chart
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, col in enumerate(pie_columns):
    ax = axes[idx // 2, idx % 2]
    counts = data[col].value_counts()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title(f'{col} Distribution')

plt.tight_layout()
plt.show()



# %%
## Analysis of specific products (related to churn or retention)
# Phone Service vs Customer Status
plt.figure(figsize=(6, 4))
sns.countplot(data, x='Phone Service', hue='Customer Status')
plt.title('Phone Service vs Customer Status')
plt.xlabel('Phone Service')
plt.ylabel('Count')
plt.legend(title='Customer Status')
plt.show()

# Internet Type vs Customer Status
plt.figure(figsize=(6, 4))
sns.countplot(data, x='Internet Type', hue='Customer Status')
plt.title('Internet Type vs Customer Status')
plt.xlabel('Internet Type')
plt.ylabel('Count')
plt.legend(title='Customer Status')
plt.xticks(rotation=20)
plt.show()

# Streaming TV vs Customer Status
plt.figure(figsize=(6, 4))
sns.countplot(data, x='Streaming TV', hue='Customer Status')
plt.title('Streaming TV vs Customer Status')
plt.xlabel('Streaming TV')
plt.ylabel('Count')
plt.legend(title='Customer Status')
plt.show()

# Streaming Movies vs Customer Status
plt.figure(figsize=(6, 4))
sns.countplot(data, x='Streaming Movies', hue='Customer Status')
plt.title('Streaming Movies vs Customer Status')
plt.xlabel('Streaming Movies')
plt.ylabel('Count')
plt.legend(title='Customer Status')
plt.show()


#%%

##  using subplots (we have to choose.)

product_features = ['Phone Service', 'Internet Type', 'Streaming TV', 'Streaming Movies']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, feature in enumerate(product_features):
    ax = axes[idx // 2, idx % 2]
    sns.countplot(data, x=feature, hue='Customer Status', ax=ax)
    ax.set_title(f'{feature} vs Customer Status')
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    ax.legend(title='Customer Status')
    ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()


#Phone Service : Most customers (overwhelmingly) are Yes, 
#However, the Customer Status distribution is similar for both Yes and No

#Internet Type vs Customer Status:
#Fiber Optic users are numerous, and the churn rate among them is also quite high
#DSL users have a much higher stay rate and a lower churn rate

# Streaming TV vs Customer Status:
#Distinctive distribution difference between Yes and No
#Streaming TV = No Customers have a higher churn rate

#Streaming Movies vs. Customer Status:
#This is the opposite trend: Customers with Streaming Movies = No have a higher churn rate.


# %%
## chi-square test
from scipy.stats import chi2_contingency
product_features = ['Phone Service', 'Internet Type', 'Streaming TV', 'Streaming Movies']

chi2_results = {}

for col in product_features:
    
    contingency_table = pd.crosstab(data[col], data['Customer Status'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    chi2_results[col] = {
        'Chi-square statistic': round(chi2, 2),
        'p-value': round(p, 5),
        'Degrees of freedom': dof
    }

for feature, result in chi2_results.items():
    print(f"--- {feature} ---")
    print(f"Chi-square: {result['Chi-square statistic']}, p-value: {result['p-value']}, DoF: {result['Degrees of freedom']}")
    print()

# Phone Service showed no significant association with customer status (p = 0.31).
# In contrast, Internet Type, Streaming TV, and Streaming Movies all showed statistically significant associations with churn (p < 0.001), 
# suggesting that customers using more data-intensive or premium content services may be more likely to churn, possibly due to unmet performance or value expectations.


# %%
