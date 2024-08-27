#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Prerequisites: Pre-trained ANN or ensemble model

# Usage Example:
# python3 predict.py 25 M PA /path/to/image.png 0.5 /path/to/trained_model.h5
# 1) 25 = age of the patient (1 to 130 range)
# 2) M = gender of the patient (M = male, F = female)
# 3) PA = X-ray view position (PA = posteroanterior, AP = anteroposterior)
# 4) /path/to/image.png = path to an x-ray image
# 5) 0.5 = likelihood threshold (0.5 will display all outcomes with >=50% chance)
# 6) /path/to/trained_model.h5 = path to the trained ANN model

# python3 script_name.py age gender x-ray_angle /path/to/image.png likelihood /path/to/trained_model


# In[1]:


import argparse
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model


# In[2]:


# Define the label columns (one-hot encoded columns)
label_columns = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]


# ### Creating a function to process input images

# In[17]:


def preprocess_image(image_path):
    '''
    Function to preprocess an input image from path
    '''
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    # Resize the image to the desired size (28x28)
    image = cv2.resize(image, (28, 28))
    
    # Normalize the image
    image = image.astype('float32') / 255.0
    
    # Expand dimensions to match the input shape of the model (batch_size, height, width, channels)
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    
    return image


# ### Getting input data to make a prediction

# In[20]:


def valid_age(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 130:
        raise argparse.ArgumentTypeError(f"Age must be between 1 and 130. You entered {ivalue}.")
    return ivalue

def valid_gender(value):
    if value.upper() not in ['M', 'F']:
        raise argparse.ArgumentTypeError(f"Gender must be 'M' or 'F'. You entered {value}.")
    return value.upper()

def valid_view_position(value):
    if value.upper() not in ['PA', 'AP']:
        raise argparse.ArgumentTypeError(f"View position must be 'PA' or 'AP'. You entered {value}.")
    return value.upper()

def valid_image_path(value):
    if not os.path.exists(value):
        raise argparse.ArgumentTypeError(f"The specified path does not exist: {value}")
    return value

def valid_model_path(value):
    if not os.path.exists(value):
        raise argparse.ArgumentTypeError(f"The specified model path does not exist: {value}")
    return value

def make_prediction(model, age, gender, view_position, image_path, likelihood_threshold):
    '''
    Take user input and output likelihood predictions for each of the 14 possible outcomes.
    '''
    # Preprocess the image
    image_array = preprocess_image(image_path)

    # Prepare the input data
    single_input_age = np.array([[age]])
    single_input_gender = np.array([[gender]])
    single_input_view_position = np.array([[view_position]])

    # Make a prediction
    prediction = model.predict([single_input_age, single_input_gender, single_input_view_position, image_array])

    # Format the prediction output
    formatted_prediction = {label: prob for label, prob in zip(label_columns, prediction[0])}

    # Print the prediction values that exceed the threshold
    print(f"Prediction (values > {likelihood_threshold}):")
    predictions_above_threshold = False
    for label, prob in formatted_prediction.items():
        if prob > likelihood_threshold:
            print(f"{label}: {prob:.4f}")
            predictions_above_threshold = True

    if not predictions_above_threshold:
        print(f"No predictions exceed the threshold of {likelihood_threshold}.")

    # Ask the user if they want to see the full list
    show_full_list = input("Would you like to see the full list of predictions? (y/n): ").strip().lower()

    if show_full_list == 'y':
        print("\nFull Prediction List:")
        for label, prob in formatted_prediction.items():
            print(f"{label}: {prob:.4f}")


# In[ ]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions based on user input.")
    parser.add_argument("age", type=valid_age, help="Enter patient age (1-130)")
    parser.add_argument("gender", type=valid_gender, help="Enter patient gender (M/F)")
    parser.add_argument("view_position", type=valid_view_position, help="Enter view position (PA/AP)")
    parser.add_argument("image_path", type=valid_image_path, help="Enter the path to the image")
    parser.add_argument("likelihood_threshold", type=float, help="Enter the likelihood threshold")
    parser.add_argument("model_path", type=valid_model_path, help="Enter the path to the model")

    args = parser.parse_args()

    # Convert gender and view_position to numerical values
    gender = 0 if args.gender == 'M' else 1
    view_position = 0 if args.view_position == 'PA' else 1

    # Load the model
    loaded_model = load_model(args.model_path)

    # Call the function with parsed arguments
    make_prediction(loaded_model, args.age, gender, view_position, args.image_path, args.likelihood_threshold)

