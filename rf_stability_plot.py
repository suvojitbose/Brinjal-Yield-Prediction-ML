import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, cross_validate
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data from Excel
xls = pd.ExcelFile('AMMI FINAL DATA (1).xlsx')
df1 = pd.read_excel(xls, sheet_name=0)
df2 = pd.read_excel(xls, sheet_name=1)

# Clean TSS column (convert to string, replace spaces, convert to float)
df2['TSS'] = df2['TSS'].astype(str).str.replace(' ', '').astype(float)
df1['TSS'] = df2['TSS'].values

# Define Features and Target
features = ['PH', 'PB', 'DF', 'NFP', 'FL', 'FD', 'FW', 'ASC', 'AN', 'FP', 'TSS']
target = 'FYP'

for col in features + [target]:
    if df1[col].dtype == object:
        df1[col] = df1[col].astype(str).str.replace(' ', '').astype(float)

X = df1[features]
y = df1[target]

# 2. Initialize Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 3. Repeated K-Fold Cross-Validation setup
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

# 4. Evaluate Models
cv_results = cross_validate(rf_model, X, y, cv=rkf, return_estimator=True)
rf_estimators = cv_results['estimator']

# 5. Extract and Plot RF Feature Importance Stability
importances = np.array([tree.feature_importances_ for tree in rf_estimators])
mean_importances = np.mean(importances, axis=0)
std_importances = np.std(importances, axis=0)

feat_imp_df = pd.DataFrame({
    'Trait': features,
    'Mean Importance': mean_importances,
    'Std Dev': std_importances
}).sort_values(by='Mean Importance', ascending=False)

# 6. Plotting
plt.figure(figsize=(10, 6), dpi=300)
sns.barplot(x='Mean Importance', y='Trait', data=feat_imp_df, palette="viridis")

# Adding error bars manually
plt.errorbar(x=feat_imp_df['Mean Importance'], y=range(len(feat_imp_df)), 
             xerr=feat_imp_df['Std Dev'], fmt='none', c='black', capsize=5)

plt.title('Random Forest Feature Importance Stability\n(Across 50 Cross-Validation Folds)')
plt.xlabel('Mean Decrease in Impurity (± SD)')
plt.ylabel('Morphological & Biochemical Traits')
plt.tight_layout()

# Save the plot
plt.savefig('RF_Stability_Plot.png')
print("Plot generated successfully!")