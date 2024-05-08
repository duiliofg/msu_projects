# Import necessary libraries
import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, f1_score
from imblearn.over_sampling import RandomOverSampler

# Function to load a hyperspectral image from a given file path
def load_hyperspectral_image(path):
    with rasterio.open(path) as src:
        image_data = src.read()  # Read all bands of the hyperspectral image
        metadata = src.meta      # Retrieve metadata including spatial transformations
    return image_data, metadata

# Function to extract training data using labeled geospatial vector data (geodataframe)
def extract_training_data(hyperspectral_image, metadata, geodataframe, label_column):
    training_data = []
    labels = []
    transform = metadata['transform']
    for index, feature in geodataframe.iterrows():
        geom = feature.geometry
        # Generate a mask for the given geometry to extract pixels within the polygon
        mask = rasterio.features.geometry_mask([geom], out_shape=(metadata['height'], metadata['width']),
                                               transform=transform, invert=True)
        # Extract indices where mask is True and collect the corresponding hyperspectral data
        pixel_indices = np.argwhere(mask)
        for (y, x) in pixel_indices:
            pixel_value = hyperspectral_image[:, y, x]
            training_data.append(pixel_value)
            labels.append(feature[label_column])
    return np.array(training_data), np.array(labels)

# Function to create a custom Convolutional Neural Network model
def create_custom_cnn(input_shape, num_classes, filters1, kernel_size1, filters2, kernel_size2):
    inputs = Input(shape=input_shape)
    # First convolutional block
    x = Conv2D(filters1, kernel_size1, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    # Second convolutional block
    x = Conv2D(filters2, kernel_size2, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.7)(x)
    # Dense layers after flattening the data
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    # Compile the model with an optimizer, loss function, and metrics
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load hyperspectral and geospatial vector data
raster_path = "/home/g98z895/projects/albedo/input/20240412_sodankyla_masked_2.tiff"
gdf_path = "/home/g98z895/projects/albedo/input/samples_sodankyla_polygons.gpkg"
hyperspectral_data, metadata = load_hyperspectral_image(raster_path)
gdf = gpd.read_file(gdf_path)

# Extract training data
training_data, labels = extract_training_data(hyperspectral_data, metadata, gdf, 'type')

if len(training_data) > 0:
    num_samples, num_bands = training_data.shape[0], training_data.shape[1]
    training_data = training_data.reshape(num_samples, 1, 1, num_bands)
    
    # Apply random oversampling to address class imbalance in the training dataset
    ros = RandomOverSampler(random_state=0)
    training_data_reshaped = training_data.reshape(num_samples, -1)
    training_data_resampled, labels_resampled = ros.fit_resample(training_data_reshaped, labels)
    training_data_resampled = training_data_resampled.reshape(-1, 1, 1, num_bands)
    
    num_classes = np.max(labels_resampled) + 1

    # Initialize DataFrame to store the results of various model configurations
    results_df = pd.DataFrame(columns=['Filters1', 'Kernel Size1', 'Filters2', 'Kernel Size2', 'Epochs', 'Precision', 'F1 Score'])

    # Iterate over combinations of filter sizes, kernel sizes, and training epochs
    filters_options_layer = [[16,32], [32, 64], [64,128], [128,256]]  # Different configurations of filters
    kernel_sizes_layers = [[(1,1),(3,3)],[(2,2),(4,4)],[(3,3),(5,5)]]
    epochs_options = [35,50,75]

    for filters in filters_options_layer:
        for kernel_size in kernel_sizes_layers:
            for epochs in epochs_options:
                try:
                    model = create_custom_cnn((1, 1, num_bands), num_classes, filters[0], kernel_size[0], filters[1], kernel_size[1])
                    model.fit(training_data_resampled, labels_resampled, epochs=epochs, validation_data=(training_data_resampled, labels_resampled))
                    predictions = model.predict(training_data_resampled).argmax(axis=-1)
                    precision = precision_score(labels_resampled, predictions, average='macro')
                    f1 = f1_score(labels_resampled, predictions, average='macro')
                    # Add results to the DataFrame
                    results_df = pd.concat([results_df, pd.DataFrame([{'Filters1': filters[0],
                                                                       'Kernel Size1': kernel_size[0],
                                                                       'Filters2': filters[1],
                                                                       'Kernel Size2': kernel_size[1],
                                                                       'Epochs': epochs,
                                                                       'Precision': precision,
                                                                       'F1 Score': f1}])], ignore_index=True)
                except Exception as e:
                    # Handle possible exceptions during model training or evaluation
                    results_df = pd.concat([results_df, pd.DataFrame([{'Filters1': filters[0],
                                                                       'Kernel Size1': kernel_size[0],
                                                                       'Filters2': filters[1],
                                                                       'Kernel Size2': kernel_size[1],
                                                                       'Epochs': epochs,
                                                                       'Precision': 0.0,
                                                                       'F1 Score': 0.0}])])
    # Display the results sorted by F1 Score
    print(results_df.sort_values(by='F1 Score', ascending=False))
else:
    print("Insufficient data for training.")

# Save the results to a CSV file
results_df.to_csv('/home/g98z895/projects/albedo/output/test.csv')

