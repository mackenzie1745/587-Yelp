
import numpy as np
import json
import math
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


f = open("labeled_reviews_1to200.json", encoding='UTF-8')
data_food = {}
data_service = {}
food_label = []
food_train = [] 
service_label = [] 
service_train = []

for line in f:
    B = json.loads(line)
    data_food["review_id"] = B["review_id"]
    data_service["review_id"] = B["review_id"]
    data_food["text"] = B["text"]
    data_service["text"] = B["text"]
    food_train.append(B["text"])
    service_train.append(B["text"])

    if B["label"] == "b":
    	data_food["label"] = 1
    	food_label.append(1)
    	data_service["label"] = 1
    	service_label.append(1)

    elif B["label"] == "n":
    	data_food["label"] = 0
    	food_label.append(0)
    	data_service["label"] = 0
    	service_label.append(0)

    elif B["label"] == "f":
    	data_food["label"] = 1
    	food_label.append(1)
    	data_service["label"] = 0
    	service_label.append(0)

    else:
    	data_food["label"] = 0
    	food_label.append(0)
    	data_service["label"] = 1
    	service_label.append(1)
f.close()

def classification(train,label,train_percent):
	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(train)
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	y = label


	train_len = int(X_train_tfidf.shape[0]*train_percent)
	X_train = X_train_tfidf[0:train_len, : ]
	y_train = y[0:train_len]
	X_test = X_train_tfidf[train_len: , : ]
	y_test = y[train_len: ]

	print("training data number: ", train_len)
	print("training data dimension: ", X_train.shape)
	print("test data number: ", X_test.shape)

	clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr').fit(X_train, y_train)

	result = clf.predict(X_test)
	for i in range(len(result)):
		print("prediction: ",result[i], "label: ", y_test[i])

	### print("test data prediction prob: ", clf.predict_proba(X_test))
	print("test data prediction accuracy: ", clf.score(X_test, y_test), "\n")

classification(food_train,food_label,0.8)
classification(service_train,service_label,0.8)
















