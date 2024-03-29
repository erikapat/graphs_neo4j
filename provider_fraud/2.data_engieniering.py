
import pandas as pd


pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 999)

neo4j_path = 'C:/Users/epgonz1/.Neo4jDesktop/relate-data/dbmss/dbms-cd4711a2-8187-492e-b9af-b4e1fffe4125/import/'
neo4j_path = 'data/'
# Load data
beneficiaries = pd.read_csv(neo4j_path + "Train_Beneficiarydata-1542865627584.csv")
inpatients = pd.read_csv(neo4j_path + "Train_Inpatientdata-1542865627584.csv")
outpatients = pd.read_csv(neo4j_path + "Train_Outpatientdata-1542865627584.csv")

labels = pd.read_csv('data/Train-1542865627584.csv')

# organize data
df1 = inpatients[['BeneID', 'ClaimID', 'Provider', 'InscClaimAmtReimbursed', 'AttendingPhysician', 'DeductibleAmtPaid']]
df2 = outpatients[['BeneID', 'ClaimID', 'Provider', 'InscClaimAmtReimbursed', 'AttendingPhysician', 'DeductibleAmtPaid']]
df3 = beneficiaries[['BeneID', 'Gender', 'Race', 'NoOfMonths_PartACov', 'NoOfMonths_PartBCov',
                     'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt',
                     'OPAnnualDeductibleAmt']]


df = pd.concat([df1, df2])
df = df.set_index('Provider').join(labels.set_index('Provider')).reset_index().set_index('BeneID').join(df3.set_index('BeneID')).reset_index()
df['PotentialFraud'] = df['PotentialFraud'].replace("No", 0).replace("Yes", 1).astype(int)
# Providers: start with 10 & AttendingPhysician: starts with 2
df['Provider'] = df['Provider'].str.removeprefix("PRV").astype(int) + 1_000_000
df['AttendingPhysician'] = df['AttendingPhysician'].str.removeprefix("PHY").fillna(0).astype(int) + 2_000_000

# General data
print('saving data')
df.to_parquet('output_data/claim_features.parquet')
# print(df.head())
# Data for Neo4j

# Nodes
# This to claim level
df_select = df[['BeneID', 'ClaimID', 'PotentialFraud', 'AttendingPhysician', 'Provider']]
physician_nodes_df = df_select[['AttendingPhysician']]
physician_nodes_df.rename(columns={'AttendingPhysician': 'physician_id'}, inplace=True)
physician_nodes_df['category_id'] = 'AttendingPhysician'
physician_nodes_df = physician_nodes_df.drop_duplicates()
provider_nodes_df = df_select[['Provider']]
provider_nodes_df.rename(columns={'Provider': 'provider_id'}, inplace=True)
provider_nodes_df['category_id'] = 'Provider'
provider_nodes_df = provider_nodes_df.drop_duplicates()

print(physician_nodes_df.dtypes)
print(provider_nodes_df.dtypes)
physician_nodes_df.to_csv('output_data/physician_nodes_df.csv', index=False)
provider_nodes_df.to_csv('output_data/provider_nodes_df.csv', index=False)

# Edges
# Provider -> works with -> AttendingPhysician -> attending to -> BeneId

#edges
df_work_with = df[['Provider', 'AttendingPhysician', 'ClaimID', 'PotentialFraud']]
df_work_with.rename(columns={'Provider': 'provider_id', 'AttendingPhysician': 'physician_id'}, inplace=True)
df_work_with.to_csv('output_data/df_work_with.csv', index=False)
print(df_work_with.dtypes)