# Take the scikit learn breast cancer dataset with 30 dimensions. 
#~Reduce the dimensions to 2 using PCA. 
#Then using Naive Bayes model, classify the cancer of a patient into malignant or benign.



from flask import Flask, request, jsonify, render_template
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load and preprocess the data
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensions to 2 using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Train the Naive Bayes classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Create the Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Simple form for data input

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = [float(x) for x in request.form.values()]
        if len(features) != 2:
            return jsonify({'error': 'Please provide exactly 2 features.'})

        # Reshape input for prediction
        input_features = np.array(features).reshape(1, -1)
        prediction = classifier.predict(input_features)
        result = 'Malignant' if prediction[0] == 0 else 'Benign'

        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)