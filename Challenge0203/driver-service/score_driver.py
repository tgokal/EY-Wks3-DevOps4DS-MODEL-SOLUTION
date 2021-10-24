import json
import joblib
import numpy as np
from azureml.core.model import Model

# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = Model.get_model_path('driver-training')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    # Get a prediction from the model
    predictions = model.predict(data)
    # Get the corresponding classname for each prediction (0 or 1)
    classnames = ['no claim', 'claim']
    predicted_classes = []
    probabilities = [prediction for prediction in predictions]
    for probability in probabilities:
        if probability < 0.5:
            predicted_classes.append(classnames[0])
        else:
            predicted_classes.append(classnames[1])

    # Return the predictions
    return (probabilities, predicted_classes)
