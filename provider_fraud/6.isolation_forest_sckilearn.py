
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, average_precision_score
import matplotlib.pyplot as plt
import shap

pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 999)

# Read Parquet file into DataFrame
data = pd.read_parquet('output_data/df_enriched.parquet')


classes = data['PotentialFraud'].to_numpy()
print('The Class Imbalance: %s' % Counter(classes))


X = data.drop(['PotentialFraud', 'BeneID', 'ClaimID', 'Provider', 'AttendingPhysician'], axis=1)
X = X.fillna(0)
y = data.PotentialFraud
from sklearn.ensemble import IsolationForest
Iforest = IsolationForest(max_samples=100,
                          random_state=1111,
                         contamination=0.01,
                         max_features=1.0,
                         n_estimators=100,
                         verbose=1,
                         n_jobs=-1)
Iforest.fit(X)

# predict
y_pred = Iforest.predict(X)
y_pred_adjusted = [1 if x == -1 else 0 for x in y_pred]
print(sum(y_pred_adjusted))

from sklearn.metrics import precision_recall_fscore_support
print(precision_recall_fscore_support(y, y_pred_adjusted, average='macro'))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred_adjusted)
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.savefig('fig/confusion.png')
plt.show()

contam_dict = {
    0.01 : (0.7036742817915103, 0.508547813612243, 0.40334318267142727, 5582),
    0.05 : (0.6198364321162996, 0.5241313290000188, 0.4525769562515801, 27911),
    0.10 : (0.596811795091418, 0.5369370509167648, 0.49358618374695934, 55821)
}

contamdf = pd.DataFrame.from_dict(contam_dict, orient='index').reset_index()
contamdf.columns =['contamination','precision', 'recall', 'fscore',"false_postives"]
print(contamdf)

# https://medium.com/mlthinkbox/anomaly-detection-with-isolation-forest-in-scikit-learn-99417dcc3971

# ------------------------------------------------------------------

# Interpretability

explainer = shap.Explainer(Iforest.predict, X)
shap_values = explainer(X)