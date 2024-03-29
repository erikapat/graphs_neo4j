
import pandas as pd
import networkx as nx
from utils.plot_graphs import feat_cor
from utils.graph_metrics import findCommunities
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics


from sklearn.ensemble import RandomForestClassifier


from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# Speed up hyperparameter tuning
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

# Read Parquet file into DataFrame
df = pd.read_parquet('output_data/claim_features.parquet')
# -------------------------------------------------------------------------------------------------------------

source = 'Provider'
target = 'AttendingPhysician'

G = nx.from_pandas_edgelist(df, source=source, target=target)
# plot
# plot_graphs(df, source, target)

# Enrichment of features -------------------------------------------------------------------------------------

# CENTRALITY
# defining a dictionary of graph features
nodes_info_dict = {
  # 'closeness_centrality': nx.closeness_centrality,
  'eigenvector_centrality': nx.eigenvector_centrality_numpy,
  'pagerank': nx.pagerank
}

columns_with_node_infos = ['degree'] + list(nodes_info_dict.keys())

nodes_info = pd.DataFrame.from_dict(dict(nx.degree(G)), orient='index').rename(columns = {0 : 'degree'}).reset_index()

# computing graph features for each node
for info, fun in nodes_info_dict.items():
    temp = pd.DataFrame.from_dict(fun(G), orient='index').rename(columns = {0 : info}).reset_index()
    nodes_info = nodes_info.merge(temp, on='index')

nodes_info = nodes_info.rename(columns = {'index': 'Physician'})

# adding graph features to the dataframe
df_enriched = df.merge(nodes_info, left_on = 'Provider',
                           right_on='Physician', how='left').drop('Physician', axis=1)
df_enriched.rename(columns = {k:'Provider_'+k for k in columns_with_node_infos}, inplace = True)

df_enriched = df_enriched.merge(nodes_info, left_on = 'AttendingPhysician',
                           right_on='Physician', how='left').drop('Physician', axis=1)
df_enriched.rename(columns = {k:'AttendingPhysician_'+k for k in columns_with_node_infos}, inplace = True)

# ------------------------------------------------------------------------------------------------------------------
# Plot relations between Potential fraud & graph features

feat_cor(df_enriched)

# ------------------------------------------------------------------------------------------------------------------

# Community detection
G = findCommunities(G)

# -----------------------------------------------------------------------------------------------------------------

#I = G.subgraph(list(H.nodes()))
#drawNetwork(I)

df_communities = pd.DataFrame([[k, v] for k, v in nx.get_node_attributes(G, 'community').items()], columns=["AttendingPhysician", "AttendingPhysician_cluster"])
df_enriched = df_enriched.set_index('AttendingPhysician').join(df_communities.set_index('AttendingPhysician'), how="left", rsuffix='_comm').reset_index()

print('saving data')
df_enriched.to_parquet('output_data/df_enriched.parquet')

'''
I tested 3 scenarios that vary in terms of features used for the training of the models:

Scenario 1 — Baseline: information about the claim and the patient (cf. section 1.2.)

Scenario 2 — Baseline and graph’s features: These features include the 4 metrics described above (cf. section 1.3.): the degree of the nodes representing physicians, their closeness centrality coefficient, eigenvector centrality, and PageRank.

Scenario 3 — Baseline, graph’s features, and detected communities: The algorithms tested are those explained above (cf. section 2.): the Louvain method, InfoMap, and RandomWalk.
'''

classes = df_enriched['PotentialFraud'].to_numpy()

print('The Class Imbalance: %s' % Counter(classes))

# Splitting feature data from label data
X, y = df_enriched[['InscClaimAmtReimbursed',
                    'DeductibleAmtPaid',
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
                    'AttendingPhysician_cluster']], df_enriched['PotentialFraud']

print("Original shapes: ", "X:", X.shape, " y:", y.shape)

X = X.fillna(0)

 # Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=69)
print(f"Shapes after splitting:\n\nX_train: {X_train.shape}, y_train: {y_train.shape}\
      \nX_test: {X_test.shape}, y_test: {y_test.shape}")

accuracy = []
feature_names = ["Baseline", "Graph Features", "Graph Features with Community Detection"]
features = [
    # Baseline features
    ['InscClaimAmtReimbursed',
     'DeductibleAmtPaid',
     'Gender',
     'Race',
     'NoOfMonths_PartACov',
     'NoOfMonths_PartBCov',
     'IPAnnualReimbursementAmt',
     'IPAnnualDeductibleAmt',
     'OPAnnualReimbursementAmt',
     'OPAnnualDeductibleAmt'],
    # Baseline + Graph Features
    ['InscClaimAmtReimbursed',
     'DeductibleAmtPaid',
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
    'AttendingPhysician_pagerank'],
    # Baseline + Graph Features + Community Detection
    ['InscClaimAmtReimbursed',
     'DeductibleAmtPaid',
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
     'AttendingPhysician_cluster']
]

hyper_parameter_grids_RFC = [
    { # Grid 1: No regularization
      "randomforestclassifier__criterion": ['gini'],
      "randomforestclassifier__max_depth": [10, 20, 50, 100, 250, 300, 500],
      "randomforestclassifier__min_samples_split": [2, 3, 5, 10, 20, 30],
    },
    { # Grid 2: L2 regularization
      "randomforestclassifier__criterion": ['entropy'],
      "randomforestclassifier__max_depth": [10, 20, 50, 100, 250, 300, 500],
      "randomforestclassifier__min_samples_split": [2, 3, 5, 10, 20, 30],

    },
]

pipeline_RFC = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=69))


for feature in features:
    print("*" * 100)
    print("# Tuning hyper-parameters for accuracy")
    print("*" * 100)
    print()

    # This performs gridsearch, evaluating each set of hyper-parameters using k-fold
    # cross validation.
    clf = HalvingGridSearchCV(pipeline_RFC, hyper_parameter_grids_RFC, scoring="accuracy", cv=4, n_jobs=-1)

    X_train_subset = X_train[feature]
    X_test_subset = X_test[feature]

    clf.fit(X_train_subset, y_train)

    acc = round(clf.best_estimator_.score(X_test_subset, y_test) * 100, 2)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Best score on development set:")
    print()

    print(f"Accuracy: {acc}")
    accuracy.append(acc)

    # ---------------------------------------------------------------------------

    # define metrics
    y_pred_proba = clf.best_estimator_.steps[1][1].predict_proba(X_test[feature])[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    '''
    model_metrics = {
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'precision':  metrics.precision_score(y_true, y_pred),
        'recall':  metrics.recall_score(y_true, y_pred),
        'f1-recall':  metrics.recall_score(y_true, y_pred),
    }
    
    model_metrics
    '''
    # create ROC curve
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)

plt.savefig('fig/roc.png')
plt.show()

# ---------------------------------------------------------------------------
bar_data = pd.DataFrame(
    dict(
        labels=list(X_train.columns),
        feature_importance=clf.best_estimator_.steps[1][1].feature_importances_
    )
)

bar_data = bar_data.sort_values('feature_importance', ascending=False)
bar_data.plot(x="labels", y="feature_importance", kind="bar")
plt.savefig('fig/feature_importance.png')
plt.show()

# References
# https://towardsdatascience.com/fraud-detection-with-graph-analytics-2678e817b69e
# https://jiaxiangbu.github.io/anti_fraud_practice/datacamp.html
# https://www.kaggle.com/code/matthewmaddock/graph-features-random-forest-networkx
# https://pub.towardsai.net/insurance-fraud-detection-with-graph-analytics-91d10c5e5ec9

