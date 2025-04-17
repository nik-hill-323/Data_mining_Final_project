#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

#%%

# Analyze the relationship between Contract type and Customer Status.
cleaned_df = pd.read_csv("cleaned_data.csv") 

# Calculate Customer Status Ratio (%) by Contract Type
contract_churn = pd.crosstab(cleaned_df['Contract'], cleaned_df['Customer Status'], normalize='index') * 100
contract_churn

#%%
# Churn vs Stayed Ratio by Contract Type
plt.figure(figsize=(8, 6))
contract_churn[['Churned', 'Stayed']].plot(kind='bar', stacked=True, colormap='Set2')
plt.title('Customer Status by Contract Type')
plt.ylabel('Percentage (%)')
plt.xlabel('Contract Type')
plt.xticks(rotation=0)
plt.legend(title='Customer Status')
plt.tight_layout()
plt.show()

# Month-to-month customers have a high Churned rate and a low Stayed rate
# Most customers with one-year or two-year contracts are Stayed

#%%
# Chi-square test
contingency_table = pd.crosstab(cleaned_df['Contract'], cleaned_df['Customer Status'])
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-square statistic: {chi2:.2f}")
print(f"p-value: {p:.5f}")
print(f"Degrees of freedom: {dof}")

# p < 0.001 -> Contract type and churn are statistically related

# %%
# “How well does contract type predict churn?
# Is contract type still important when combined with other variables?
# y= churn/ x= contract (additional: tenure, montly charge, age, streaming services, internet type? )

# logisitc regression 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# y
cleaned_df['Churn'] = cleaned_df['Customer Status'].apply(lambda x: 1 if x == 'Churned' else 0)
# x
features = ['Contract', 'Tenure in Months', 'Monthly Charge', 'Age']
X = cleaned_df[features]
y = cleaned_df['Churn']

categorical_features = ['Contract']
numerical_features = ['Tenure in Months', 'Monthly Charge', 'Age']

# (OneHotEncoder(c) + StandardScaler(n))
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop='first'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

#Defining the entire modeling pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred)

print(report)
# Overall accuracy 81%, churn prediction recall 63%, which is quite good
#Long-term contract customers are significantly less likely to churn, as confirmed by model coefficients and prediction performance



# %%
# coefficient

#Extract encoded variable names
ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
encoded_feature_names = ohe.get_feature_names_out(['Contract'])
final_feature_names = list(encoded_feature_names) + numerical_features

coefficients = pipeline.named_steps['classifier'].coef_[0]


coef_df = pd.DataFrame({
    'Feature': final_feature_names,
    'Coefficient': np.round(coefficients, 4)
}).sort_values(by='Coefficient', key=abs, ascending=False)


print("\nLogistic Regression Coefficients:")
print(coef_df)

#Contract_Two Year(-2.48) A two-year contract strongly reduces the probability of churn (based on month-to-month)
#Contract_One Year (-1.48) A one-year contract also reduces the probability of churn
#Tenure in Months (-0.98) The longer the subscription period, the lower the probability of churn
#Monthly Charge (+0.74) The higher the charge, the higher the probability of churn
#Age(+0.31) The older the customer, the slightly higher the probability of churn
# %%
