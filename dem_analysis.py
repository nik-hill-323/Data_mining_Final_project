#%%
import pandas as pd

df = pd.read_csv("Dataset/cleaned_churn_data.csv")
# %%
import matplotlib.pyplot as plt
import pandas as pd

gender_counts = pd.crosstab(df['Gender'], df['Customer Status'])

gender_counts.plot(kind='bar', stacked=True, colormap='Pastel1')
plt.title('Customer Status by Gender')
plt.ylabel('Number of Customers')
plt.xlabel('Gender')
plt.xticks(rotation=0)
plt.legend(title='Customer Status')
plt.tight_layout()
plt.show()

# %%
married_counts = pd.crosstab(df['Married'], df['Customer Status'])

married_counts.plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title('Customer Status by Marital Status')
plt.ylabel('Number of Customers')
plt.xlabel('Married')
plt.xticks(rotation=0)
plt.legend(title='Customer Status')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(8, 6))
for status in df['Customer Status'].unique():
    subset = df[df['Customer Status'] == status]
    plt.hist(subset['Age'], bins=20, alpha=0.5, label=status)

plt.title('Age Distribution by Customer Status')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# %%
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Customer Status', y='Number of Dependents', palette='Set2')
plt.title('Dependents by Customer Status')
plt.tight_layout()
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Define age bins
bins = [18, 25, 35, 45, 55, 65, 75, 100]
labels = ['18–24', '25–34', '35–44', '45–54', '55–64', '65–74', '75+']
df['Age Bracket'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Step 2: Calculate churn rate per age bracket
churn_by_age = df.groupby('Age Bracket')['Customer Status'].value_counts(normalize=True).unstack()
churn_percentage = churn_by_age.get('Churned', 0) * 100  # percentage

# Step 3: Plot
churn_percentage.plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Percentage of Customers Who Churned by Age Bracket')
plt.ylabel('% Churned')
plt.xlabel('Age Bracket')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Create age brackets
bins = [18, 25, 35, 45, 55, 65, 75, 100]
labels = ['18–24', '25–34', '35–44', '45–54', '55–64', '65–74', '75+']
df['Age Bracket'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Step 2: Group by Age Bracket and Gender and calculate churn rate
grouped = df.groupby(['Age Bracket', 'Gender'])['Customer Status'].value_counts(normalize=True).unstack()
churn_pct = grouped.get('Churned', 0).unstack() * 100  # rows: Age Bracket, columns: Gender

# Step 3: Plot
ax = churn_pct.plot(kind='bar', figsize=(10, 6), colormap='Set2', edgecolor='black')

plt.title('Churn Percentage by Age Bracket and Gender')
plt.ylabel('% Churned')
plt.xlabel('Age Bracket')
plt.xticks(rotation=45)
plt.ylim(0, 100)
plt.legend(title='Gender')
plt.tight_layout()
plt.show()

# %%
