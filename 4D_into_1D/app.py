# Take iris flowers data set and reduce the 4D into 1D using PCA. 
#Then train your model and predict a new flower with given measurements.

from flask import Flask, request, jsonify, render_template
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the iris dataset
iris = load_iris()
X = iris.data  # Features (4D)
y = iris.target  # Labels

# Reduce the dataset to 1D using PCA
pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Train a model (Random Forest Classifier)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Create a Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input features from request
    try:
        data = [float(x) for x in request.form.values()]
        # Transform the input using PCA
        input_features = pca.transform([data])
        # Make a prediction
        prediction = clf.predict(input_features)
        result = iris.target_names[prediction[0]]
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
