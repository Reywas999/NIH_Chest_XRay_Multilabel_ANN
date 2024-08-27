#!/usr/bin/env python
# coding: utf-8

# Eventually this should be tidied up...

# ### Necessary imports

# In[1]:

# Open the text file with the required libraries
with open('requirements.txt', 'r') as file:
    libraries = file.read()

# Execute the import statements
exec(libraries)


# ### Setting the likelihood threshold
# ##### If likelihood of outcome > threshold in a prediction, it will be printed

# In[2]:


likelihood_threshold = 0.25
destination_dir = r"X_Ray_NN\All_images" # Destination for the X-ray images
# Train and test sets are split such that individuals who appear more than once are not found in both sets.
test_list_path = r"X_Ray_NN\Split_Train_Val\test_list.txt" # Test set path
train_list_path = r"X_Ray_NN\Split_Train_Val\train_val_list.txt" # Train set path
best_trained_model = r"X_Ray_NN\trained_model.h5" # Path to the best trained model (if one exists, else will save here)
# Define the label columns (one-hot encoded columns)
label_columns = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]


# # NIH CXR8 Chest X-Rays
# NIH Chest X-ray Dataset of 14 Common Thorax Disease Categories:\
# \
# https://nihcc.app.box.com/v/ChestXray-NIHCC/file/371647823217
# \
# \
# 1, Atelectasis; \
# 2, Cardiomegaly; \
# 3, Effusion; \
# 4, Infiltration; \
# 5, Mass; \
# 6, Nodule; \
# 7, Pneumonia; \
# 8, Pneumothorax; \
# 9, Consolidation; \
# 10, Edema; \
# 11, Emphysema; \
# 12, Fibrosis; \
# 13, Pleural_Thickening; \
# 14, Hernia\
# \
# Meta data for all images (Data_Entry_2017_v2020.csv): Image Index, Finding Labels, Follow-up #,
# Patient ID, Patient Age, Patient Gender, View Position, Original Image Size and Original Image
# Pixel Spacing.\
# \
# Bounding boxes for ~1000 images (BBox_List_2017.csv): Image Index, Finding Label,
# Bbox[x, y, w, h]. [x y] are coordinates of each box's topleft corner. [w h] represent the width and
# height of each box.\
# \
# Two data split files (train_val_list.txt and test_list.txt) are provided. Images in the ChestX-ray
# dataset are divided into these two sets on the patient level. All studies from the same patient will
# only appear in either training/validation or testing set.

# In[3]:


def image_retrieval():
    '''
    Retrieving and compiling all the NIH CXR8 Chest X-Ray images if the destination directory does not
    already exist.
    '''
    # Check if the destination directory already exists
    if os.path.exists(destination_dir):
        print(f"{destination_dir} already exists. Skipping download and extraction.")
        return
    # Download the 56 zip files in Images_png in batches
    # URLs for the zip files
    # 112,120 frontal-view chest X-ray PNG images in 1024*1024 resolution
    links = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
    ]

    for idx, link in enumerate(links):
        fn = 'images_%02d.tar.gz' % (idx+1)
        print('downloading '+fn+'...')
        urllib.request.urlretrieve(link, fn)  # download the zip file

    print("Download complete. Please check the checksums")
    
    # Define the paths
    source_dir = r"X_Ray_NN\images"

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Loop through each .tar.gz file in the source directory
    for file_name in os.listdir(source_dir):
        if file_name.endswith(".tar.gz"):
            file_path = os.path.join(source_dir, file_name)

            # Open the tar.gz file
            with tarfile.open(file_path, "r:gz") as tar:
                # Extract all files to a temporary directory
                temp_dir = os.path.join(source_dir, "temp")
                tar.extractall(path=temp_dir)

                # Move all images to the destination directory
                images_dir = os.path.join(temp_dir, "images")
                for image_name in os.listdir(images_dir):
                    image_path = os.path.join(images_dir, image_name)
                    shutil.move(image_path, destination_dir)

                # Clean up the temporary directory
                shutil.rmtree(temp_dir)

    print("All images have been extracted to:", destination_dir)


# In[4]:


def get_evidence(path=r"X_Ray_NN\Data_Entry_2017_v2020.csv"):
    '''
    Reading in the labels and additional evidence as a pandas DF
    Encoding gender, view position, and one hot encoding the label values
    '''
    # Path to the CSV file with labels and further input data
    file_path = path

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Change the values in the "Patient Gender" column
    df['Patient Gender'] = df['Patient Gender'].map({'M': 0, 'F': 1})

    # Change the values in the "View Position" column
    df['View Position'] = df['View Position'].map({'PA': 0, 'AP': 1})

    # One-hot encode the "Finding Labels" column
    # Split the labels by '|' and create a set of unique labels, excluding "No Finding"
    labels = set()
    df['Finding Labels'].str.split('|').apply(labels.update)
    labels.discard('No Finding')
    labels = sorted(labels)

    # Create a new DataFrame for the one-hot encoded labels
    one_hot_encoded_labels = pd.DataFrame(0, index=df.index, columns=labels)

    # Populate the one-hot encoded DataFrame
    for index, row in df.iterrows():
        if 'No Finding' in row['Finding Labels']:
            continue  # Skip if "No Finding" is present
        for label in row['Finding Labels'].split('|'):
            one_hot_encoded_labels.at[index, label] = 1

    # Concatenate the original DataFrame with the one-hot encoded labels
    df = pd.concat([df, one_hot_encoded_labels], axis=1)

    # Drop the original "Finding Labels" column
    df.drop(columns=['Finding Labels'], inplace=True)
    
    return df


# In[5]:


def image_processing(path = destination_dir):
    '''
    Reformatting the images as arrays fit for ANNs
    '''
    # Initialize a list to store the image arrays
    image_arrays = []

    # List all files in the directory
    for img_name in os.listdir(path):
        # Generate the image path
        img_path = os.path.join(path, img_name)

        # Load the image in grayscale mode
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue  # Skip if the image is not loaded properly

        # Resize the image to the desired size (e.g., 28x28)
        image = cv2.resize(image, (28, 28))

        # Normalize the image
        image = image.astype('float32') / 255.0

        # Expand dimensions to match the input shape of the model (batch_size, height, width, channels)
        image = np.expand_dims(image, axis=-1)  # Add channel dimension

        # Convert image to array
        image_array = img_to_array(image)

        # Add the image array to the list
        image_arrays.append((img_name, image_array))
        
    return image_arrays


# In[6]:


def merge_arrays(arrays, data):
    '''
    Appending the image arrays to the evidence/labels dataframe, then saving the DF as a CSV file.
    '''
    # Create a new DataFrame for the image arrays
    image_df = pd.DataFrame(image_arrays, columns=['Image Index', 'Image Array'])

    # Merge the original DataFrame with the image DataFrame on the "Image Index" column
    df = pd.merge(df, image_df, on='Image Index', how='left')
    
    # Saving the DF as a CSV file
    df.to_csv(r"X_Ray_NN\processed_data.csv", index=False)
    
    return df


# In[ ]:





# In[7]:


def str_to_array(array_str):
    '''
    Function to convert string representation of array to actual array for formatting an imported CSV
    '''
    # Extract all numbers from the string
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", array_str)
    # Convert extracted numbers to float
    numbers = list(map(float, numbers))
    # Reshape the list into the desired array shape
    array = np.array(numbers).reshape((28, 28, 1))  # Adjust the shape as needed
    return array


# In[8]:


def reading_csv_data(path = r"X_Ray_NN\processed_data.csv"):
    '''
    Reading in and formatting the CSV file for future use.
    '''
    df = pd.read_csv(path)
    # Apply the function to the "Image Array" column
    df['Image Array'] = df['Image Array'].apply(str_to_array)
    return df


# In[9]:


def get_data():
    csv_path = r"X_Ray_NN\processed_data.csv"
    # Check if the destination directory already exists
    if os.path.exists(csv_path):
        print(f"{csv_path} already exists. Reading in and formatting the data.")
        df = reading_csv_data()
        return df
    else:
        image_retrieval()
        df = get_evidence()
        image_arrays = image_processing()
        df = merge_arrays(image_arrays, df)
        return df


# ### Creating test and training datasets, split into evidence (x) and label (y) DFs

# In[10]:


def preprocess_data(df, test_list_path = test_list_path, train_list_path = train_list_path):
    '''
    Split the df into testing and training sets
    Splitting training/testing sets into evidence (x) and labels (y)
    Only keep three of the columns to use for analysis (for now)
    '''
    # Read the text files into lists
    with open(test_list_path, 'r') as file:
        test_list = file.read().splitlines()

    with open(train_list_path, 'r') as file:
        train_val_list = file.read().splitlines()

    # Filter the DataFrame based on the lists
    test_data = df[df['Image Index'].isin(test_list)]
    train_data = df[df['Image Index'].isin(train_val_list)]

    # Split the training DataFrame into evidence and labels
    x_train = train_data.drop(columns=label_columns)
    y_train = train_data[label_columns]

    # Split the testing DataFrame into evidence and labels
    x_test = test_data.drop(columns=label_columns)
    y_test = test_data[label_columns]
    
    # For now, I would like to only keep three of the columns to use for analysis.
    # Define the columns to keep
    columns_to_keep = ['Patient Age', 'Patient Gender', 'View Position', 'Image Array']

    # Filter the x_train and x_test DataFrames
    x_train = x_train[columns_to_keep]
    x_test = x_test[columns_to_keep]
    
    return x_train, y_train, x_test, y_test


# ### Creating the model

# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


class CustomEarlyStopping(Callback):
    '''
    Custom callback to stop training early if there is not a change of greater than 0.0055 after
    5 epochs, and if accuracy is decreasing.
    '''
    def __init__(self, patience=5, min_delta=0.0055):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_weights = None
        self.best_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get("accuracy")
        if current_accuracy is None:
            return

        if current_accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = current_accuracy
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
                print(f"\nEpoch {epoch + 1}: early stopping triggered")


# In[12]:


def build_model(hp):
    '''
    Building and compiling an ANN sequential model, this model is not called... 
    Its simply a test model
    '''
    # Define the input layers
    age_input = Input(shape=(1,), name='Patient_Age')
    gender_input = Input(shape=(1,), name='Patient_Gender')
    view_position_input = Input(shape=(1,), name='View_Position')
    image_input = Input(shape=(28, 28, 1), name='Image_Array')

    # Process the image input
    x = Flatten()(image_input)
    x = Dense(hp.Int('units_1', min_value=32, max_value=512, step=32), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1))(x)
    x = Dense(hp.Int('units_2', min_value=32, max_value=512, step=32), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1))(x)

    # Concatenate all inputs
    concatenated = Concatenate()([age_input, gender_input, view_position_input, x])

    # Add a few dense layers
    x = Dense(hp.Int('units_3', min_value=32, max_value=512, step=32), activation='relu')(concatenated)
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('dropout_3', min_value=0.0, max_value=0.5, step=0.1))(x)
    x = Dense(hp.Int('units_4', min_value=32, max_value=512, step=32), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('dropout_4', min_value=0.0, max_value=0.5, step=0.1))(x)

    # Output feature vector instead of final classification
    feature_output = Dense(64, activation='relu', name='Feature_Output')(x)
    
    # Output layer with 14 units (one for each label) and sigmoid activation for multi-label classification
    output = Dense(14, activation='sigmoid', name='Output')(x)

    # Define the model
    model = Model(inputs=[age_input, gender_input, view_position_input, image_input], outputs=output)
        
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# In[13]:


def find_best_model(x_train, y_train, x_test, y_test):
    '''
    This is the ANN model called in the combined function. Searches through different hyperparameters
    to find the best possible combination (from the parameters randomly tested)
    '''
    # Convert DataFrame columns to NumPy arrays
    x_train_age = np.array(x_train['Patient Age']).reshape(-1, 1)
    x_train_gender = np.array(x_train['Patient Gender']).reshape(-1, 1)
    x_train_view_position = np.array(x_train['View Position']).reshape(-1, 1)
    x_train_images = np.array([img for img in x_train['Image Array']])

    y_train = np.array(y_train)

    # Ensure the image arrays have the correct shape
    x_train_images = np.array([img.reshape(28, 28, 1) for img in x_train_images])

    # Convert DataFrame columns to NumPy arrays
    x_test_age = np.array(x_test['Patient Age']).reshape(-1, 1)
    x_test_gender = np.array(x_test['Patient Gender']).reshape(-1, 1)
    x_test_view_position = np.array(x_test['View Position']).reshape(-1, 1)
    x_test_images = np.array([img for img in x_test['Image Array']])

    y_test = np.array(y_test)

    # Ensure the image arrays have the correct shape
    x_test_images = np.array([img.reshape(28, 28, 1) for img in x_test_images])
    
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=50,
        factor=3,
        directory='my_dir',
        project_name='xray_classification'
    )
    
    tuner.search(
        [x_train_age, x_train_gender, x_train_view_position, x_train_images],
        y_train,
        epochs=50,
        validation_split=0.2,
        callbacks=[CustomEarlyStopping(patience=5, min_delta=0.0075)]
    )
    
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model


# In[ ]:





# In[14]:


def model_eval_fit_save(model, evaluate = True, fit = True, save = True):
    '''
    evaluate, fit, and save the input model, can alter the defaults for each.
    '''
    if evaluate:
        model.evaluate(
            [x_test_age, x_test_gender, x_test_view_position, x_test_images],
            y_test
        )
    if fit:
        best_model.fit(
            [x_train_age, x_train_gender, x_train_view_position, x_train_images],
            y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.2
        )
    if save:
        best_model.save(r"X_Ray_NN\trained_model.h5")


# In[15]:


def retrieve_model():
    '''
    If a model is already built, retrieve it, else retrieve and process the data, build the best model
    and then evaluate, fit, and save the model before returning it
    '''
    if os.path.exists(best_trained_model):
        print(f"{best_trained_model} already exists. Loading the model now.")
        loaded_model = tf.keras.models.load_model(best_trained_model)
        return loaded_model
    else:
        df = get_data()
        x_train, y_train, x_test, y_test = preprocess_data(df)
        best_model = find_best_model(x_train, y_train, x_test, y_test)
        model_eval_fit_save(best_model, evaluate = True, fit = True, save = True)
        return best_model


# In[16]:


loaded_model = retrieve_model()


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


def make_prediction(likelihood=likelihood_threshold):
    '''
    Take user input and output likelihood predictions for each of the 14 possible outcomes.
    '''
    # Load a model to use. This function will either load an already saved model, or will retrieve,
    # process, and load the data to build a model, build a model, then evaluate, save, and load
    # the model for use
    loaded_model = retrieve_model()
    
    # Get user input with validation
    while True:
        try:
            age = int(input("Enter patient age (1-130): "))
            if 1 <= age <= 130:
                break
            else:
                print("Please enter a valid age between 1 and 130.")
        except ValueError:
            print("Please enter a valid number for age.")

    while True:
        gender = input("Enter patient gender (M/F): ").upper()
        if gender in ['M', 'F']:
            gender = 0 if gender == 'M' else 1
            break
        else:
            print("Please enter 'M' for male or 'F' for female.")

    while True:
        view_position = input("Enter view position (PA/AP): ").upper()
        if view_position in ['PA', 'AP']:
            view_position = 0 if view_position == 'PA' else 1
            break
        else:
            print("Please enter 'PA' for posteroanterior or 'AP' for anteroposterior.")

    while True:
        image_path = input("Enter the path to the image: ")
        if os.path.exists(image_path):
            break
        else:
            print("The specified path does not exist. Please enter a valid path.")

    # Preprocess the image
    image_array = preprocess_image(image_path)

    # Prepare the input data
    single_input_age = np.array([[age]])
    single_input_gender = np.array([[gender]])
    single_input_view_position = np.array([[view_position]])

    # Make a prediction
    prediction = loaded_model.predict([single_input_age, single_input_gender, single_input_view_position, image_array])

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
        print('')
        print(f"No predictions exceed the threshold of {likelihood_threshold}.")
        print('')

    # Ask the user if they want to see the full list
    show_full_list = input("Would you like to see the full list of predictions? (y/n): ").strip().lower()

    if show_full_list == 'y':
        print("\nFull Prediction List:")
        for label, prob in formatted_prediction.items():
            print(f"{label}: {prob:.4f}")


# In[25]:


make_prediction(loaded_model)

