# commented out ipython magic to ensure python compatibility.
# ======================================================================
# bank customer churn analysis
# ======================================================================
# This script analyzes factors contributing to customer churn in a banking dataset,
# builds a predictive model, and performs customer segmentation.
# ======================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

import shap
import warnings
warnings.filterwarnings('ignore')

# For inline plotting
# %matplotlib inline

# Load the dataset
df = pd.read_csv('/content/telecom_customer_churn.csv')
print("Initial Data Preview:")
print(df.head())

# Drop columns that are not useful for analysis
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
print("Data after dropping unnecessary columns:")
print(df.head())

# Display dataset information and statistics
print("Dataset Information:")
df.info()
print("\nDataset Description:")
print(df.describe())

# Plot the churn distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Exited', data=df)
plt.title('Churn Distribution')
plt.show()

label_encoders = {}
for col in ['Geography', 'Gender']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("Data after encoding categorical variables:")
print(df.head())

# Separate features and target variable
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print the classification report to evaluate performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Create a SHAP explainer and compute SHAP values for the test set
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Display the SHAP summary plot
shap.summary_plot(shap_values, X_test)

# Before scaling, fill missing values in the DataFrame using the median
df.fillna(df.median(), inplace=True)

# For clustering, we use all features except the target variable 'Exited'
features_for_clustering = df.drop('Exited', axis=1)

# Scale the feature data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features_for_clustering)

# Use the Elbow Method to determine the optimal number of clusters
inertia = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(scaled_data)
    inertia.append(km.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(6,4))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters')
plt.show()

# --- Extended EDA and Hypothesis Testing ---

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Assume 'df' is already loaded, cleaned, and preprocessed (dropped 'RowNumber', 'CustomerId', 'Surname')
# Also assume categorical features 'Geography' and 'Gender' have been encoded.
print("Dataset Info:")
df.info()
print("\nDataset Description:")
print(df.describe())

# Plot the churn distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Exited', data=df)
plt.title('Churn Distribution')
plt.show()

# --- Correlation Analysis ---
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# --- Boxplots for Key Continuous Variables ---
# Compare Age distribution between churned and non-churned customers
plt.figure(figsize=(8,6))
sns.boxplot(x='Exited', y='Age', data=df)
plt.title("Age Distribution by Churn")
plt.show()

# Compare CreditScore distribution
plt.figure(figsize=(8,6))
sns.boxplot(x='Exited', y='CreditScore', data=df)
plt.title("CreditScore by Churn")
plt.show()

# --- Hypothesis Testing ---

# Split the data into churned and non-churned groups for continuous variables
churned = df[df['Exited'] == 1]
not_churned = df[df['Exited'] == 0]

# 1. T-test for Age difference between churned and non-churned
t_stat_age, p_value_age = stats.ttest_ind(churned['Age'], not_churned['Age'])
print("T-test for Age Difference:")
print(f"t-statistic: {t_stat_age:.3f}, p-value: {p_value_age:.3f}")
if p_value_age < 0.05:
    print("-> The difference in Age is statistically significant.\n")
else:
    print("-> The difference in Age is not statistically significant.\n")

# 2. T-test for CreditScore difference
t_stat_cs, p_value_cs = stats.ttest_ind(churned['CreditScore'], not_churned['CreditScore'])
print("T-test for CreditScore Difference:")
print(f"t-statistic: {t_stat_cs:.3f}, p-value: {p_value_cs:.3f}")
if p_value_cs < 0.05:
    print("-> The difference in CreditScore is statistically significant.\n")
else:
    print("-> The difference in CreditScore is not statistically significant.\n")

# 3. Chi-Square Test for categorical variables

# For Gender and Churn
contingency_gender = pd.crosstab(df['Gender'], df['Exited'])
chi2_gender, p_gender, dof_gender, ex_gender = stats.chi2_contingency(contingency_gender)
print("Chi-Square Test for Gender and Churn:")
print(f"Chi2: {chi2_gender:.3f}, p-value: {p_gender:.3f}")
if p_gender < 0.05:
    print("-> There is a significant association between Gender and Churn.\n")
else:
    print("-> There is no significant association between Gender and Churn.\n")

# For Geography and Churn
contingency_geo = pd.crosstab(df['Geography'], df['Exited'])
chi2_geo, p_geo, dof_geo, ex_geo = stats.chi2_contingency(contingency_geo)
print("Chi-Square Test for Geography and Churn:")
print(f"Chi2: {chi2_geo:.3f}, p-value: {p_geo:.3f}")
if p_geo < 0.05:
    print("-> There is a significant association between Geography and Churn.\n")
else:
    print("-> There is no significant association between Geography and Churn.\n")

# 4. Chi-Square Test for HasCrCard and Churn
contingency_cc = pd.crosstab(df['HasCrCard'], df['Exited'])
chi2_cc, p_cc, dof_cc, ex_cc = stats.chi2_contingency(contingency_cc)
print("Chi-Square Test for HasCrCard and Churn:")
print(f"Chi2: {chi2_cc:.3f}, p-value: {p_cc:.3f}")
if p_cc < 0.05:
    print("-> There is a significant association between HasCrCard and Churn.\n")
else:
    print("-> There is no significant association between HasCrCard and Churn.\n")

import seaborn as sns
import matplotlib.pyplot as plt

# Create a clustered heatmap for the correlation matrix
plt.figure(figsize=(12,10))
sns.clustermap(df.corr(), annot=True, cmap='viridis')
plt.title("Clustered Correlation Heatmap", pad=100)  # pad to move title above the clustermap
plt.show()

# Select a subset of features for a pairplot
selected_features = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary', 'Tenure', 'Exited']
sns.pairplot(df[selected_features], hue='Exited', palette='Set2', diag_kind='kde', markers=["o", "s"])
plt.suptitle("Pair Plot of Selected Features", y=1.02)
plt.show()

# Violin plot for Age distribution by churn
plt.figure(figsize=(8,6))
sns.violinplot(x='Exited', y='Age', data=df, palette='Pastel1')
plt.title("Age Distribution by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.show()

# Violin plot for CreditScore by churn
plt.figure(figsize=(8,6))
sns.violinplot(x='Exited', y='CreditScore', data=df, palette='Pastel2')
plt.title("CreditScore Distribution by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x='Exited', y='Age', data=df, palette='Set3', showfliers=False)
sns.swarmplot(x='Exited', y='Age', data=df, color='0.25')
plt.title("Age Distribution by Churn with Individual Data Points")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.show()

# Distribution of Balance
plt.figure(figsize=(8,6))
sns.histplot(df['Balance'], kde=True, bins=30, color='skyblue')
plt.title("Distribution of Balance")
plt.xlabel("Balance")
plt.ylabel("Frequency")
plt.show()

# Distribution of EstimatedSalary
plt.figure(figsize=(8,6))
sns.histplot(df['EstimatedSalary'], kde=True, bins=30, color='salmon')
plt.title("Distribution of Estimated Salary")
plt.xlabel("Estimated Salary")
plt.ylabel("Frequency")
plt.show()

g = sns.FacetGrid(df, col="Geography", row="Exited", margin_titles=True, height=3, aspect=1.2, palette="muted")
g.map(sns.histplot, "Age", bins=20, color="steelblue")
g.fig.suptitle("Age Distribution by Geography and Churn", y=1.05)
plt.show()

# Count plot for Geography by churn
plt.figure(figsize=(8,6))
sns.countplot(x='Geography', hue='Exited', data=df, palette='coolwarm')
plt.title("Churn Count by Geography")
plt.xlabel("Geography (Encoded)")
plt.ylabel("Count")
plt.legend(title="Exited", loc='upper right')
plt.show()

# Count plot for Gender by churn
plt.figure(figsize=(8,6))
sns.countplot(x='Gender', hue='Exited', data=df, palette='coolwarm')
plt.title("Churn Count by Gender")
plt.xlabel("Gender (Encoded)")
plt.ylabel("Count")
plt.legend(title="Exited", loc='upper right')
plt.show()

import pandas as pd

# Load dataset and drop unneeded columns
df = pd.read_csv('Churn_Modelling.csv')
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# Exclude joint people – assuming joint customers hold more than one product:
df = df[df['NumOfProducts'] == 1]

import pandas as pd

# Load the dataset
df = pd.read_csv('Churn_Modelling.csv')

# Remove name-related columns for privacy
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# Exclude joint customers by filtering for individual accounts only
# (assuming customers with NumOfProducts equal to 1 are individual accounts)
df = df[df['NumOfProducts'] == 1]

# Display information to confirm the changes
print("Data shape after removing joint people (names) and filtering individual accounts:", df.shape)
df.head()

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# List continuous features to analyze
features = ['Age', 'CreditScore', 'Tenure', 'Balance', 'EstimatedSalary']

for feature in features:
    # Boxplot: Visualize distribution by churn status
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Exited', y=feature, data=df, palette='pastel')
    plt.title(f'{feature} by Churn')
    plt.xlabel("Churn (0 = Retained, 1 = Churned)")
    plt.show()

    # Density plot: Visualize the feature’s distribution for churned and retained customers
    plt.figure(figsize=(6,4))
    sns.kdeplot(data=df, x=feature, hue='Exited', fill=True, palette='Set2', alpha=0.5)
    plt.title(f'Distribution of {feature} by Churn')
    plt.show()

    # t-Test: Determine if the mean differences between groups are statistically significant
    churned = df[df['Exited'] == 1][feature]
    retained = df[df['Exited'] == 0][feature]
    t_stat, p_val = stats.ttest_ind(churned, retained, nan_policy='omit')
    print(f"T-test for {feature}: t-statistic = {t_stat:.3f}, p-value = {p_val:.3f}")
    if p_val < 0.05:
        print("=> The difference is statistically significant.\n")
    else:
        print("=> No statistically significant difference.\n")

import numpy as np
import pandas as pd

# Simulate a ContractType column if not present (for demonstration purposes).
if 'ContractType' not in df.columns:
    np.random.seed(42)  # for reproducibility
    contract_options = ['Month-to-month', 'One-year', 'Two-year']
    df['ContractType'] = np.random.choice(contract_options, size=len(df))

# Count plot: Visualize churn counts by contract type.
plt.figure(figsize=(6,4))
sns.countplot(x='ContractType', hue='Exited', data=df, palette='pastel')
plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Count")
plt.show()

# Create a contingency table between ContractType and Churn status.
contingency = pd.crosstab(df['ContractType'], df['Exited'])

# Perform chi-square test to check association.
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"Chi-square Test (ContractType vs. Churn): chi2 = {chi2:.3f}, p-value = {p:.3f}")
if p < 0.05:
    print("=> There is a significant association between contract type and churn.\n")
else:
    print("=> No significant association between contract type and churn.\n")

# Create age groups for finer demographic analysis.
df['AgeGroup'] = pd.cut(df['Age'], bins=[20,30,40,50,60,70,80],
                        labels=['20-30', '30-40', '40-50', '50-60', '60-70', '70-80'])

# Visualize churn counts by age group.
plt.figure(figsize=(6,4))
sns.countplot(x='AgeGroup', hue='Exited', data=df, palette='cool')
plt.title("Churn by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.show()

# Visualize churn counts by gender.
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', hue='Exited', data=df, palette='Set2')
plt.title("Churn by Gender")
plt.xlabel("Gender (Encoded)")
plt.ylabel("Count")
plt.show()

# Create a contingency table and perform a chi-square test for Gender vs. Churn.
contingency_gender = pd.crosstab(df['Gender'], df['Exited'])
chi2_gender, p_gender, dof_gender, expected_gender = stats.chi2_contingency(contingency_gender)
print(f"Chi-square Test (Gender vs. Churn): chi2 = {chi2_gender:.3f}, p-value = {p_gender:.3f}")
if p_gender < 0.05:
    print("=> There is a significant association between gender and churn.")
else:
    print("=> No significant association between gender and churn.")

"""
Telecom Customer Churn Analysis - Individual Customers Only
This script analyzes factors contributing to customer churn in a telecom dataset,
focusing only on individual customers (not joint/family accounts).
"""

#%%
# ======================================================================
# 1. IMPORT LIBRARIES
# ======================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('colorblind')

#%%
# ======================================================================
# 2. LOAD AND PREPROCESS DATA
# ======================================================================
def load_and_filter_individual_customers(file_path):
    """
    Load the telecom dataset and filter to only include individual customers.
    It filters out joint accounts based on:
      - 'Multiple Lines' is not 'Yes'
      - Customers not having all three streaming services (indicating a joint/family account).
    Then, it retains only customers with status 'Churned' or 'Stayed'.
    """
    print(f"\nLoading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Original dataset size: {df.shape[0]} rows, {df.shape[1]} columns")

    # Filter for individual customers
    individual_customers = df[
        (df['Multiple Lines'] != 'Yes') &
        ~((df['Streaming TV'] == 'Yes') &
          (df['Streaming Movies'] == 'Yes') &
          (df['Streaming Music'] == 'Yes'))
    ]
    print(f"After filtering for individual customers: {individual_customers.shape[0]} rows")
    removed = df.shape[0] - individual_customers.shape[0]
    print(f"Removed {removed} joint accounts ({(removed/df.shape[0]*100):.2f}% of original data)")

    # For analysis, keep only Churned and Stayed customers (exclude Joined)
    analysis_cohort = individual_customers[
        individual_customers['Customer Status'].isin(['Churned', 'Stayed'])
    ]
    print(f"Final analysis dataset (excluding new customers): {analysis_cohort.shape[0]} rows")
    return analysis_cohort

def preprocess_data(df):
    """
    Preprocess the DataFrame by:
      - Adding a binary 'Churn' indicator column.
      - Creating Age and Tenure groups.
      - Creating a Service Count feature.
      - Creating an Average Monthly Revenue feature.
      - Handling missing values.
    """
    processed_df = df.copy()

    # Add binary churn indicator
    processed_df['Churn'] = (processed_df['Customer Status'] == 'Churned').astype(int)

    # Create Age Groups
    bins = [18, 30, 40, 50, 60, 100]
    age_labels = ['18-29', '30-39', '40-49', '50-59', '60+']
    processed_df['Age Group'] = pd.cut(processed_df['Age'], bins=bins, labels=age_labels, right=False)

    # Create Tenure Groups
    tenure_bins = [0, 12, 24, 36, 48, 60, 100]
    tenure_labels = ['0-12 months', '13-24 months', '25-36 months', '37-48 months', '49-60 months', '61+ months']
    processed_df['Tenure Group'] = pd.cut(processed_df['Tenure in Months'], bins=tenure_bins, labels=tenure_labels, right=True)

    # Create Service Count feature
    service_columns = ['Online Security', 'Online Backup', 'Device Protection Plan',
                       'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music']
    processed_df['Service Count'] = processed_df[service_columns].apply(lambda x: (x == 'Yes').sum(), axis=1)

    # Create Average Monthly Revenue feature
    processed_df['Avg Monthly Revenue'] = processed_df.apply(
        lambda x: x['Total Revenue'] / x['Tenure in Months'] if x['Tenure in Months'] > 0 else x['Monthly Charge'], axis=1
    )

    # Handle missing values
    for col in processed_df.columns:
        if processed_df[col].dtype in ['int64', 'float64']:
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
        else:
            processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])

    return processed_df

#%%
# ======================================================================
# 3. ENHANCED EXPLORATORY DATA ANALYSIS
# ======================================================================
def perform_enhanced_eda(df):
    """
    Generate a series of creative EDA visualizations.
    Expects a preprocessed DataFrame (with 'Churn' column defined).
    """
    print("\n" + "="*70)
    print(" " * 20 + "ENHANCED EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    custom_palette = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
    churn_palette = {"Stayed": "#2ECC71", "Churned": "#E74C3C"}

    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    # Create a dashboard-style overview using GridSpec
    plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.2])

    # Overall churn rate pie chart
    ax1 = plt.subplot(gs[0, 0])
    churn_rate = df['Churn'].mean() * 100
    stay_rate = 100 - churn_rate
    sizes = [stay_rate, churn_rate]
    labels = ['Stayed', 'Churned']
    explode = (0, 0.1)
    colors = ['#2ECC71', '#E74C3C']
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    ax1.set_title('Overall Churn Rate', fontsize=16, fontweight='bold', pad=20)

    # Contract type analysis (bar chart for churn rate and customer count)
    ax2 = plt.subplot(gs[0, 1])
    contract_churn = df.groupby('Contract')['Churn'].mean() * 100
    contract_counts = df['Contract'].value_counts()
    bar_positions = np.arange(len(contract_churn))
    bar_width = 0.4
    bars = ax2.bar(bar_positions, contract_churn, bar_width, color='#E74C3C', label='Churn Rate')
    ax2.set_ylabel('Churn Rate (%)', fontsize=12, color='#E74C3C')
    ax2.set_ylim(0, max(contract_churn) * 1.2)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', color='#E74C3C', fontweight='bold')
    ax2_twin = ax2.twinx()
    ax2_twin.bar([p + bar_width for p in bar_positions], contract_counts, bar_width,
                 color='#3498DB', alpha=0.6, label='Customer Count')
    ax2_twin.set_ylabel('Number of Customers', fontsize=12, color='#3498DB')
    for i, count in enumerate(contract_counts):
        ax2_twin.text(bar_positions[i] + bar_width*1.5, count + 10,
                      f'{count}', ha='center', va='bottom', color='#3498DB')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax2.set_title('Contract Type Analysis', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xticks([p + bar_width/2 for p in bar_positions])
    ax2.set_xticklabels(contract_churn.index, rotation=0)

    # Tenure analysis (line plot)
    ax3 = plt.subplot(gs[0, 2])
    df['Tenure Group'] = pd.cut(df['Tenure in Months'],
                                bins=[0, 12, 24, 36, 48, 60, df['Tenure in Months'].max()],
                                labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61+'])
    tenure_churn = df.groupby('Tenure Group')['Churn'].mean() * 100
    sns.lineplot(x=tenure_churn.index, y=tenure_churn.values, marker='o', linewidth=3,
                 color='#9B59B6', ax=ax3, markersize=10)
    for i, v in enumerate(tenure_churn):
        ax3.text(i, v + 2, f'{v:.1f}%', ha='center', color='#9B59B6', fontweight='bold')
    ax3.set_title('Churn Rate by Tenure (Months)', fontsize=16, fontweight='bold', pad=20)
    ax3.set_xlabel('Tenure Group (Months)', fontsize=12)
    ax3.set_ylabel('Churn Rate (%)', fontsize=12)
    ax3.set_ylim(0, max(tenure_churn) * 1.2)

    # Services heatmap
    ax4 = plt.subplot(gs[1, :2])
    services = ['Online Security', 'Online Backup', 'Device Protection Plan',
                'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music']
    service_churn_rates = {}
    for service in services:
        yes_churn = df[df[service] == 'Yes']['Churn'].mean() * 100
        no_churn = df[df[service] == 'No']['Churn'].mean() * 100
        service_churn_rates[service] = {'Yes': yes_churn, 'No': no_churn}
    service_churn_df = pd.DataFrame(service_churn_rates).T
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#2ECC71', '#F39C12', '#E74C3C'])
    sns.heatmap(service_churn_df, annot=True, fmt='.1f', cmap=cmap, linewidths=1,
                annot_kws={"size": 10, "weight": "bold"}, vmin=0, vmax=60, ax=ax4)
    ax4.set_title('Churn Rate (%) by Service Subscription', fontsize=16, fontweight='bold', pad=20)
    ax4.set_ylabel('')
    ax4.text(-0.1, -0.15, "Lower values (green) indicate services that effectively reduce churn",
             transform=ax4.transAxes, ha="left", fontsize=11, style='italic')

    # Payment method and demographic analysis (grouped bar chart)
    ax5 = plt.subplot(gs[1, 2])
    demographic_factors = ['Gender', 'Married', 'Payment Method']
    factor_data = []
    for factor in demographic_factors:
        factor_categories = df[factor].unique()
        for category in factor_categories:
            churn_rate = df[df[factor] == category]['Churn'].mean() * 100
            count = df[df[factor] == category].shape[0]
            factor_data.append({
                'Factor': factor,
                'Category': category,
                'Churn Rate': churn_rate,
                'Count': count
            })
    factor_df = pd.DataFrame(factor_data)
    factor_df = factor_df.sort_values(['Factor', 'Churn Rate'], ascending=[True, False])
    sns.barplot(x='Factor', y='Churn Rate', hue='Category', data=factor_df,
                palette=custom_palette, ax=ax5)
    ax5.set_title('Churn Rate by Demographics', fontsize=16, fontweight='bold', pad=20)
    ax5.set_ylim(0, max(factor_df['Churn Rate']) * 1.2)
    ax5.set_xlabel('')
    ax5.set_ylabel('Churn Rate (%)', fontsize=12)
    ax5.legend(title='', loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig('telecom_churn_dashboard.png', dpi=300, bbox_inches='tight')

    # Additional EDA plots (Customer Journey, Geographic Analysis, etc.) follow similarly...
    # For brevity, these sections are omitted here. They remain unchanged from your original code.
    # Make sure to call plt.show() if running interactively.

    print("Enhanced EDA visualizations created and saved.")
    return df  # Return the dataframe with added analytic columns

#%%
# ======================================================================
# 4. STATISTICAL TESTS
# ======================================================================
def perform_statistical_tests(df):
    """
    Perform statistical tests to validate relationships between variables and churn.
    """
    print("\n" + "="*50)
    print(" " * 15 + "STATISTICAL TESTS")
    print("=" * 50)

    categorical_vars = ['Contract', 'Internet Type', 'Age Group', 'Tenure Group',
                        'Gender', 'Married', 'Payment Method', 'Online Security',
                        'Premium Tech Support']
    print("\nChi-Square Tests for Categorical Variables:")
    for var in categorical_vars:
        contingency = pd.crosstab(df[var], df['Churn'])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        significance = "Significant" if p < 0.05 else "Not Significant"
        print(f"{var}: chi2={chi2:.2f}, p-value={p:.6f}, {significance}")

    numerical_vars = ['Age', 'Tenure in Months', 'Monthly Charge', 'Total Charges',
                      'Service Count', 'Avg Monthly Revenue']
    print("\nT-tests for Numerical Variables:")
    for var in numerical_vars:
        churned = df[df['Churn'] == 1][var]
        stayed = df[df['Churn'] == 0][var]
        t_stat, p_val = stats.ttest_ind(churned, stayed, equal_var=False)
        significance = "Significant" if p_val < 0.05 else "Not Significant"
        churned_mean = churned.mean()
        stayed_mean = stayed.mean()
        print(f"{var}: t={t_stat:.2f}, p-value={p_val:.6f}, {significance}")
        print(f"  Mean for churned: {churned_mean:.2f}, Mean for stayed: {stayed_mean:.2f}")
        print(f"  Difference: {churned_mean - stayed_mean:.2f}\n")

#%%
# ======================================================================
# 5. FEATURE ENGINEERING AND ENCODING
# ======================================================================
def prepare_features_for_modeling(df):
    """
    Prepare features for machine learning models by encoding categorical variables.
    """
    model_df = df.copy()
    cat_features = ['Contract', 'Internet Type', 'Gender', 'Married', 'Payment Method',
                    'Online Security', 'Online Backup', 'Device Protection Plan',
                    'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
                    'Streaming Music', 'Unlimited Data', 'Paperless Billing']
    num_features = ['Age', 'Number of Dependents', 'Tenure in Months',
                    'Monthly Charge', 'Total Charges', 'Service Count',
                    'Avg Monthly Revenue']
    encoded_df = model_df.copy()
    label_encoders = {}
    for col in cat_features:
        if col in encoded_df.columns:
            le = LabelEncoder()
            encoded_df[col] = encoded_df[col].fillna('Unknown')
            encoded_df[col + '_Encoded'] = le.fit_transform(encoded_df[col])
            label_encoders[col] = le
    encoded_cat_features = [col + '_Encoded' for col in cat_features if col in encoded_df.columns]
    model_features = num_features + encoded_cat_features
    X = encoded_df[model_features].copy()
    y = encoded_df['Churn'].copy()
    return X, y, model_features, label_encoders

#%%
# ======================================================================
# 6. MODELING - LOGISTIC REGRESSION
# ======================================================================
def build_logistic_regression_model(X, y):
    """
    Build and evaluate a logistic regression model.
    """
    print("\n" + "="*50)
    print(" " * 10 + "LOGISTIC REGRESSION MODEL")
    print("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    }
    print("\nPerforming grid search for hyperparameter tuning...")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='f1')
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 score: {cv_scores.mean():.4f}")

    best_model.fit(X_train_scaled, y_train)

    y_pred = best_model.predict(X_test_scaled)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Stayed', 'Churned'],
                yticklabels=['Stayed', 'Churned'])
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig('logistic_regression_cm.png')

    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('logistic_regression_roc.png')

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(best_model.coef_[0])
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance.head(10).to_string(index=False))

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('Top 15 Features by Importance (Logistic Regression)', fontsize=14)
    plt.xlabel('Absolute Coefficient Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('logistic_regression_feature_importance.png')

    return best_model, feature_importance

#%%
# ======================================================================
# 7. MODELING - RANDOM FOREST
# ======================================================================
def build_random_forest_model(X, y):
    """
    Build and evaluate a Random Forest classifier.
    """
    print("\n" + "="*50)
    print(" " * 10 + "RANDOM FOREST MODEL")
    print("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    print("\nPerforming grid search for hyperparameter tuning...")
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    y_pred = best_rf.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Stayed', 'Churned'],
                yticklabels=['Stayed', 'Churned'])
    plt.title('Confusion Matrix (Random Forest)', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig('random_forest_cm.png')

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_rf.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance.head(10).to_string(index=False))

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('Top 15 Features by Importance (Random Forest)', fontsize=14)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('random_forest_feature_importance.png')

    return best_rf, feature_importance

#%%
# ======================================================================
# 8. CLUSTERING ANALYSIS
# ======================================================================
def perform_clustering(df):
    """
    Perform K-means clustering to identify customer segments.
    """
    print("\n" + "="*50)
    print(" " * 10 + "CUSTOMER SEGMENTATION WITH K-MEANS")
    print("=" * 50)

    clustering_features = ['Age', 'Tenure in Months', 'Monthly Charge',
                           'Service Count', 'Avg Monthly Revenue']
    cluster_data = df[clustering_features].copy()

    for col in cluster_data.columns:
        cluster_data[col] = cluster_data[col].fillna(cluster_data[col].median())

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)

    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters', fontsize=12)
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    plt.title('Elbow Method for Optimal Number of Clusters', fontsize=14)
    plt.tight_layout()
    plt.savefig('kmeans_elbow.png')

    optimal_k = 4  # Choose based on the elbow curve
    print(f"\nSelected optimal number of clusters: {optimal_k}")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = clusters

    # Group by clusters for analysis
    cluster_analysis = df_with_clusters.groupby('Cluster').agg({
        'Churn': 'mean',
        'Age': 'mean',
        'Tenure in Months': 'mean',
        'Monthly Charge': 'mean',
        'Service Count': 'mean',
        'Avg Monthly Revenue': 'mean',
        'Customer ID': 'count'
    }).reset_index()
    cluster_analysis['Percentage'] = (cluster_analysis['Customer ID'] / cluster_analysis['Customer ID'].sum() * 100).round(2)
    cluster_analysis = cluster_analysis.rename(columns={'Churn': 'Churn Rate', 'Customer ID': 'Customers'})
    cluster_analysis['Churn Rate'] = (cluster_analysis['Churn Rate'] * 100).round(2)

    print("\nCluster Analysis:")
    print(cluster_analysis.to_string(index=False))

    categorical_vars = ['Contract', 'Internet Type', 'Gender', 'Married', 'Payment Method']
    print("\nCategorical Variable Distribution by Cluster:")
    for var in categorical_vars:
        print(f"\n{var} Distribution:")
        distribution = pd.crosstab(df_with_clusters['Cluster'], df_with_clusters[var], normalize='index').round(4) * 100
        print(distribution)

    features_2d = ['Tenure in Months', 'Monthly Charge']
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df_with_clusters[features_2d[0]], df_with_clusters[features_2d[1]],
                          c=df_with_clusters['Cluster'], cmap='viridis', alpha=0.6, s=80)
    centers = kmeans.cluster_centers_
    centers_orig = scaler.inverse_transform(centers)
    plt.scatter(centers_orig[:, clustering_features.index(features_2d[0])],
                centers_orig[:, clustering_features.index(features_2d[1])],
                c='red', marker='X', s=200, alpha=0.8)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(features_2d[0], fontsize=12)
    plt.ylabel(features_2d[1], fontsize=12)
    plt.title('Customer Segments Visualization', fontsize=14)
    plt.tight_layout()
    plt.savefig('cluster_visualization.png')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Cluster', y='Churn Rate', data=cluster_analysis, palette='viridis')
    plt.title('Churn Rate by Cluster', fontsize=14)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Churn Rate (%)', fontsize=12)
    for i, row in cluster_analysis.iterrows():
        plt.text(i, row['Churn Rate'] + 1, f"n={row['Customers']}\n({row['Percentage']}%)", ha='center')
    plt.tight_layout()
    plt.savefig('cluster_churn_rates.png')



    return df_with_clusters, cluster_analysis

#%%
# ======================================================================
# 9. RECOMMENDATIONS FUNCTION
# ======================================================================
def generate_recommendations(log_reg_importance, rf_importance, cluster_analysis, df):
    """
    Generate business recommendations based on analysis results
    """
    print("\n" + "="*50)
    print(" "*15 + "BUSINESS RECOMMENDATIONS")
    print("="*50)

    #%%
    # 1. Contract-based strategies
    month_to_month_churn = df[df['Contract'] == 'Month-to-Month']['Churn'].mean() * 100

    print("\n1. Contract-Based Retention Strategies:")
    print(f"   - Month-to-month customers have a {month_to_month_churn:.2f}% churn rate")
    print("   - Implement incentives for longer-term contracts:")
    print("     * Offer discounts for 1-year or 2-year commitments")
    print("     * Create special upgrade offers available only to contract customers")
    print("     * Provide early upgrade options for contract customers")

    #%%
    # 2. Service package recommendations
    no_security_churn = df[df['Online Security'] == 'No']['Churn'].mean() * 100
    with_security_churn = df[df['Online Security'] == 'Yes']['Churn'].mean() * 100

    print("\n2. Service Package Optimization:")
    print(f"   - Customers without Online Security churn at {no_security_churn:.2f}%")
    print(f"   - Customers with Online Security churn at only {with_security_churn:.2f}%")
    print("   - Recommended actions:")
    print("     * Bundle security features into base packages")
    print("     * Create promotional campaigns highlighting security benefits")
    print("     * Offer free trials of security services to at-risk customers")

#%%
    # 3. Targeted retention for high-risk clusters
    high_risk_cluster = cluster_analysis.sort_values('Churn Rate', ascending=False).iloc[0]

    print("\n3. Targeted Retention for High-Risk Segment:")
    print(f"   - Cluster {high_risk_cluster['Cluster']} has {high_risk_cluster['Churn Rate']:.2f}% churn rate")
    print(f"   - This segment represents {high_risk_cluster['Percentage']}% of customers")
    print("   - Key characteristics of this segment:")
    print(f"     * Average Age: {high_risk_cluster['Age']:.1f} years")
    print(f"     * Average Tenure: {high_risk_cluster['Tenure in Months']:.1f} months")
    print(f"     * Average Monthly Charge: ${high_risk_cluster['Monthly Charge']:.2f}")
    print("   - Recommended targeted interventions:")
    print("     * Proactive outreach before contract renewal")
    print("     * Personalized retention offers based on usage patterns")
    print("     * Enhanced customer service touchpoints")

    #%%
    # 4. New customer onboarding improvements
    new_customer_churn = df[df['Tenure Group'] == '0-12 months']['Churn'].mean() * 100

    print("\n4. Improved New Customer Onboarding:")
    print(f"   - New customers (0-12 months) have a {new_customer_churn:.2f}% churn rate")
    print("   - Recommendations to improve early experience:")
    print("     * Implement a structured onboarding program for first 90 days")
    print("     * Provide dedicated support contacts for new customers")
    print("     * Schedule check-in calls at 30, 60, and 90-day milestones")
    print("     * Create early-warning system to identify at-risk new customers")

    #%%
    # 5. Competitive response strategy
    competitor_reasons = df[df['Churn'] == 1]['Churn Reason'].value_counts()
    competitor_churn = sum([
        count for reason, count in competitor_reasons.items()
        if 'competitor' in reason.lower()
    ])
    competitor_percentage = competitor_churn / df['Churn'].sum() * 100

    print("\n5. Competitive Response Strategy:")
    print(f"   - {competitor_percentage:.2f}% of churned customers left for competitor reasons")
    print("   - Recommended competitive responses:")
    print("     * Implement regular competitive price/offering analysis")
    print("     * Create a 'win-back' program with special offers for recently churned customers")
    print("     * Develop a price-match or service-match guarantee for at-risk customers")
    print("     * Highlight unique service benefits in customer communications")

#%%
# ======================================================================
# 10. MAIN EXECUTION FUNCTION
# ======================================================================
def main():
    """
    Main function to execute the entire analysis pipeline
    """
    print("\n" + "="*70)
    print(" "*15 + "TELECOM CUSTOMER CHURN ANALYSIS")
    print(" "*17 + "(INDIVIDUAL CUSTOMERS ONLY)")
    print("="*70)

    # 1. Load and preprocess data
    df = load_and_filter_individual_customers('telecom_customer_churn.csv')
    df_processed = preprocess_data(df)

    # 2. Perform enhanced exploratory data analysis
    df_with_analytics = perform_enhanced_eda(df_processed)

    # 3. Perform statistical tests
    perform_statistical_tests(df_processed)

    # 4. Prepare features for modeling
    X, y, model_features, label_encoders = prepare_features_for_modeling(df_processed)

    # 5. Build and evaluate logistic regression model
    log_reg_model, log_reg_importance = build_logistic_regression_model(X, y)

    # 6. Build and evaluate random forest model
    rf_model, rf_importance = build_random_forest_model(X, y)

    # 7. Perform customer segmentation
    df_with_clusters, cluster_analysis = perform_clustering(df_processed)

    # 8. Generate recommendations
    generate_recommendations(log_reg_importance, rf_importance, cluster_analysis, df_processed)

    print("\n" + "="*70)
    print(" "*25 + "ANALYSIS COMPLETE")
    print("="*70)
    print("\nResults and visualizations saved to the current directory.")