#Take iris flowers data set and reduce the 4D data into 2D using PCA. 
#Then train your model and predict a new flower with given measurements.

from flask import Flask, request, jsonify, render_template
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Reduce dimensions from 4D to 2D using PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Train the model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_reduced, y)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from form
        data = [float(x) for x in request.form.values()]
        if len(data) != 4:
            return jsonify({"error": "Please provide exactly 4 feature values."})

        # Apply PCA transformation
        input_features = np.array(data).reshape(1, -1)
        input_reduced = pca.transform(input_features)

        # Predict the class
        prediction = clf.predict(input_reduced)
        predicted_class = iris.target_names[prediction[0]]

        return jsonify({"predicted_class": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
