import numpy as np

def predict_yield(model, temperature, rainfall):
    input_data = np.array([[temperature, rainfall]])
    return model.predict(input_data)[0]