
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, average_precision_score
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 999)

# Read Parquet file into DataFrame
data = pd.read_parquet('output_data/df_enriched.parquet')

#fillnulls
data = data.drop('DeductibleAmtPaid', axis = 1)


# check missing data
print('Missing data -----------------------------------------------')
print(data.isnull().sum())
# check duplicated data
print('Duplicated data --------------------------------------------')
print(data.drop_duplicates().count() -  data.shape[0])
print(data.head())



classes = data['PotentialFraud'].to_numpy()
print('The Class Imbalance: %s' % Counter(classes))

# Splitting feature data from label data
X, y = data[['InscClaimAmtReimbursed',
                    #'DeductibleAmtPaid',
                    'Gender',
                    'Race',
                    'NoOfMonths_PartACov',
                    'NoOfMonths_PartBCov',
                    'IPAnnualReimbursementAmt',
                    'IPAnnualDeductibleAmt',
                    'OPAnnualReimbursementAmt',
                    'OPAnnualDeductibleAmt',
                    'Provider_degree',
                    #  'Provider_closeness_centrality',
                    'Provider_eigenvector_centrality',
                    'Provider_pagerank',
                    'AttendingPhysician_degree',
                    #  'AttendingPhysician_closeness_centrality',
                    'AttendingPhysician_eigenvector_centrality',
                    'AttendingPhysician_pagerank',
                    'AttendingPhysician_cluster',
                    'PotentialFraud'
             ]], data['PotentialFraud']

print("Original shapes: ", "X:", X.shape, " y:", y.shape)

X = X.fillna(0)

 # Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=69)
print(f"Shapes after splitting:\n\nX_train: {X_train.shape}, y_train: {y_train.shape}\
      \nX_test: {X_test.shape}, y_test: {y_test.shape}")

# ------------------------------------------------------------------------------------------

## Import necessary libraries


# Define the original DataFrame

# Define the columns of interest for the Isolation Forest model
numeric_columns = data.select_dtypes(include=['float64', 'uint8', 'int64', 'int32']).columns.tolist()
data_list = data[numeric_columns]
print(data_list.shape)

# Define the scoring function (e.g., Average Precision)
scorer = make_scorer(average_precision_score)

# Define the hyperparameter dictionary

param_grid = {
    'n_estimators': [10, 20, 5],
    'max_samples': [50, 60],
    'contamination': [0.1, 0.01],
    'max_features': [0.1, 0.5],
    'bootstrap': [True],
    'random_state': [42],
    'warm_start': [True]
}


# Create the Isolation Forest model
model = IsolationForest()

# Search for the best combination of hyperparameters using GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=scorer)
grid_search.fit(data_list)


# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the Isolation Forest model with the best hyperparameters
best_model = IsolationForest(**best_params)
best_model.fit(data_list)

# Predict the outliers
outliers = best_model.predict(data_list)

# Add the outlier column to the original DataFrame
data_list["outlier"] = outliers

# Add the text columns back to the DataFrame
text_columns = data.select_dtypes(include=['object']).columns.tolist()
for col in text_columns:
    data_list[col] = data[col]

print(data_list.groupby('outlier')['AttendingPhysician'].count())
print(data_list.groupby('outlier').count())
# ----------------------------------------------------------------------------------------------------

from sklearn.preprocessing import MinMaxScaler

# Normalize the 'amount' and 'step' variables
scaler = MinMaxScaler()
data_normalized = data_list.copy()
data_normalized[['InscClaimAmtReimbursed']] = scaler.fit_transform(data_list[['InscClaimAmtReimbursed']])

# Create a scatter plot of all variables with outliers highlighted
plt.figure(figsize=(12, 8))

# Variables to include in the scatter plot
variables = ['PotentialFraud', 'InscClaimAmtReimbursed']

# Plot of normal transactions
plt.scatter(data_normalized[data_normalized['outlier'] == -1][variables[0]],
            data_normalized[data_normalized['outlier'] == -1][variables[1]],
            color='red', label='Outlier', alpha=0.5)

# Plot of outliers
plt.scatter(data_normalized[data_normalized['outlier'] == 1][variables[0]],
            data_normalized[data_normalized['outlier'] == 1][variables[1]],
            color='blue', label='Normal', alpha=0.7)

# Configure other features for the scatter plot
plt.xlabel(variables[0])
plt.ylabel(variables[1])
plt.title('Isolation Forest - Outlier Detection')

# Add legend to the plot
plt.legend()
plt.savefig('fig/isolation_forest.png')
# Show the plot
plt.show()
