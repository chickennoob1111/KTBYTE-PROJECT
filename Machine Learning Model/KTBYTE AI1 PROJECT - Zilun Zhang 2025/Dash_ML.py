
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask,request,jsonify
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier



df = pd.read_csv('New_data.csv')
# df=df.drop_duplicates()
df=df.dropna()
X = df.drop('State', axis=1)
y = df['State']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)


DecisionTree=DecisionTreeClassifier(splitter='best',random_state=757433)
DecisionTree.fit(X_train, y_train)
print("Decision Tree",DecisionTree.score(X_test, y_test))

Forest=RandomForestClassifier(n_estimators=100,random_state=757433)
Forest.fit(X_train, y_train)
print("Random Forest",Forest.score(X_test, y_test))

Logistic=LogisticRegression(max_iter=10000,random_state=757433)
Logistic.fit(X_train, y_train)
print("Logistic Regression",Logistic.score(X_test, y_test))

Kmean=KMeans(n_clusters=1000, random_state=757433)
Kmean.fit(X_train)
print("KMeans",Kmean.score(X_test, y_test))

Knn=KNeighborsClassifier(n_neighbors=10)
Knn.fit(X_train, y_train)
print("KNN",Knn.score(X_test, y_test))

models=[DecisionTree, Forest, Logistic, Kmean, Knn]

# Polynomial=PolynomialFeatures(degree=10,interaction_only=True)
# X_train_poly=Polynomial.fit_transform(X_train)
# Linear=LinearRegression()
# Linear.fit(X_train_poly,y_train)




# plt.figure(figsize=(24, 16))
# plot_tree(DecisionTree, filled=True, rounded=True, feature_names=df.drop('State', axis=1).columns,class_names=df['State'].unique().astype(str))
# plt.show()

transfer=Flask(__name__)

@transfer.route('/predict', methods=['POST'])
def predict():
    data=request.get_json()
    input_list=data['input']
    model=int(data['model'])
    if not input_list:
        return jsonify({'error': 'No input data provided'}), 400
    if len(input_list) != len(X.columns):
        return jsonify({'error': 'Invalid input data'}), 400
    try:
        input_array=np.array(input_list).reshape(1, -1)
        prediction=models[model].predict(input_array)
        predictions = [model.predict(input_array) for model in models]
        for model_name, model_prediction in zip(["Decision Tree", "Random Forest", "Logistic Regression", "KMeans", "KNN"], predictions):
            print(f"{model_name}: {model_prediction}")
        return jsonify({'prediction': prediction.tolist()}), 200
    except:
        return jsonify({'error': 'Prediction failed'}), 500

# @transfer.route('/train',methods=['POST'])
# def train():
#     data=request.get_json()
#     input_list=data['input']
#     if not input_list:
#         return jsonify({'error': 'No input data provided'}), 400
#     if len(input_list) != len(df.columns):
#         return jsonify({'error': 'Invalid input data'}), 400
#     try:
#         df.add(input_list, ignore_index=True)
#         X = df.drop('State', axis=1)
#         y = df['State']

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#         DecisionTree.fit(X_train, y_train)
#         return jsonify({'prediction': 'Training started'}), 200
#     except:
#         return jsonify({'error': 'Failed to retrieve data'}), 500



if __name__ == '__main__':
    transfer.run(debug=True,use_reloader=False, port=2482)





