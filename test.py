import requests as re
from bs4 import BeautifulSoup
import numpy as np
from joblib import load
import feature_extraction as fe  # Ensure your features module is imported
from statistics import mode  # Majority vote

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Load the model
model_path = './model/phishdefender_model9.pkl'
model = load(model_path)

def ensemble_predict(models, vector):
    predictions = [model.predict(vector)[0] for model in models]
    # Return the majority vote
    return mode(predictions)

def test_url(url):
    try:
        # Send a GET request to the URL with SSL/TLS verification enabled
        response = re.get(url, verify=True, timeout=20)

        # Check if the response is successful
        if response.status_code != 200:
            print(f"HTTP connection is not successful for the URL: {url}")
            return

        # Parse the webpage content using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract features from the parsed webpage
        vector = fe.create_vector(soup)

        # Convert the feature vector to a 2D numpy array
        vector = np.array(vector).reshape(1, -1)

        # Predict using the ensemble of models
        result = ensemble_predict(model, vector)

        # Display prediction results
        if result == 0:
            print("This web page seems legitimate!")
        else:
            print("Attention! This web page is a potential PHISHING!")

    except re.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except re.exceptions.RequestException as e:
        print("--> Error while making the request:", e)




if __name__ == '__main__':
    url = input("Enter the URL to test: ")
    test_url(url)
    print("DONE")
