import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Reference https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
# Reference https://www.kaggle.com/overload10/income-prediction-on-uci-adult-dataset

pd.set_option('display.max_columns', 1000)

# load data
headers = ["age", "workclass", "fnlwgt", "education", "education-num",
           "marital-status", "occupation", "relationship", "race", "sex",
           "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
data_set = pd.read_csv('balanced_dataset.csv', header=None, names=headers, sep=',', na_values=["?"], engine='python')
# test_set = pd.read_csv('adult.test', header=None, names=headers, sep=',\s', na_values=["?"], skiprows=1,
#                        engine='python')
print('Row count for dataset is:', data_set.shape[0])
# print('Row count for test set is:', test_set.shape[0])

# correlation analysis, drop the some columns
# data = pd.get_dummies(data_set)
# corr_matrix = data.corr()
# corr_matrix["income_>50K"].sort_values(ascending=False).to_csv("corr.txt")

# drop the columns according to the correlation results
data_set.drop(labels=['fnlwgt', 'native-country', 'workclass', 'education', 'occupation'], axis=1, inplace=True)
print(data_set.head(10))

# data clean
data_set.dropna(axis=0, how='any', inplace=True)  # drop all rows that have any missing values
print('Row count for dataset after dropping is:', data_set.shape[0])

# map features to numerical value
data_set['sex'] = data_set['sex'].map({'Male': 1, 'Female': 0}).astype(int)
# income_map = {'<=50K': 0, '>50K': 1}
# data_set['income'] = data_set['income'].map(income_map).astype(int)
rel_map = {'Unmarried': 0, 'Wife': 1, 'Husband': 2, 'Not-in-family': 3, 'Own-child': 4, 'Other-relative': 5}
data_set['relationship'] = data_set['relationship'].map(rel_map)
race_map = {'White': 0, 'Amer-Indian-Eskimo': 1, 'Asian-Pac-Islander': 2, 'Black': 3, 'Other': 4}
data_set['race'] = data_set['race'].map(race_map)
data_set['marital-status'] = data_set['marital-status'].replace(['Divorced', 'Married-spouse-absent', 'Never-married',
                                                                 'Separated', 'Widowed'], 'Single')
data_set['marital-status'] = data_set['marital-status'].replace(['Married-AF-spouse', 'Married-civ-spouse'], 'Couple')
data_set['marital-status'] = data_set['marital-status'].map({'Couple': 0, 'Single': 1})
# As the major set of capital-gain/loss is 0
data_set.loc[(data_set['capital-gain'] > 0), 'capital-gain'] = 1
data_set.loc[(data_set['capital-gain'] == 0, 'capital-gain')] = 0
data_set.loc[(data_set['capital-loss'] > 0), 'capital-loss'] = 1
data_set.loc[(data_set['capital-loss'] == 0, 'capital-loss')] = 0
print(data_set.head(10))

# split the training set and test set
X = data_set.drop(['income'], axis=1)
y = data_set['income']

split_size = 0.2

# Creation of Train and Test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=22)

print('-' * 40)
print("Train dataset: {0}{1}".format(X_train.shape, y_train.shape))
print("Test dataset: {0}{1}".format(X_test.shape, y_test.shape))

# Subsample the imbalanced data through under-sampling
# RandomUnderSampler = RandomUnderSampler()
# X_train, y_train = RandomUnderSampler.fit_sample(X_train, y_train)
# X_test, y_test = RandomUnderSampler.fit_sample(X_test, y_test)
# print("Balanced Train dataset: {0}{1}".format(X_train.shape, y_train.shape))
# print("Balanced Test dataset: {0}{1}".format(X_test.shape, y_test.shape))


# Normalization, as Multi-layer Perceptron is sensitive to feature scaling
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
mlp = MLPClassifier(hidden_layer_sizes=(9, 9, 9), max_iter=500)
mlp.fit(X_train, y_train)

prediction = mlp.predict(X_test)
print('-' * 40)
print('Accuracy score:')
print(accuracy_score(y_test, prediction))
print('-' * 40)
print('Confusion Matrix:')
print(confusion_matrix(y_test, prediction))
print('-' * 40)
print('Classification Matrix:')
print(classification_report(y_test, prediction))
