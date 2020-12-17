from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

headers = ["age", "workclass", "fnlwgt", "education", "education-num",
           "marital-status", "occupation", "relationship", "race", "sex",
           "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
data_set = pd.read_csv('adult.data', header=None, names=headers, sep=',\s', na_values=["?"], engine='python')
income_map = {'<=50K': 0, '>50K': 1}
data_set['income'] = data_set['income'].map(income_map).astype(int)
X = data_set.drop(['income'], axis=1)
y = data_set['income']

RandomUnderSampler = RandomUnderSampler()
X, y = RandomUnderSampler.fit_sample(X, y)

print("Balanced Train dataset: {0}{1}".format(X.shape, y.shape))

result = pd.concat([X, y], axis=1)
# print(result.head())
result.to_csv('balanced_dataset.csv',index=False, header=False)
