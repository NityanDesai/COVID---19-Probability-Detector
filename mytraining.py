import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import json

# Train Test Splitting
def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    tss = int(len(data) * ratio)
    tsi = shuffled[:tss]
    tri = shuffled[tss:]
    return data.iloc[tri], data.iloc[tsi]


if __name__ == '__main__':
	# Read the Data
	d = pd.read_csv('data.csv')
	train, test = data_split(d, 0.2)
	X_train = train[['fever', 'bodyPain', 'age', 'runnyNose', 'breathDiff']]
	X_test = test[['fever', 'bodyPain', 'age', 'runnyNose', 'breathDiff']]

	Y_train = train[['infectProbablity']].to_numpy().reshape(2000, )
	Y_test = test[['infectProbablity']].to_numpy().reshape(499, )

	clf = LogisticRegression()
	clf.fit(X_train, Y_train)

	file = open('model.pkl', 'wb')
	json.dump(clf, file)
	file.close()