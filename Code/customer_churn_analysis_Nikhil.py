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

