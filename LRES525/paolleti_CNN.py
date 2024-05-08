# Import necessary libraries
import rasterio
from tensorflow.keras.utils import to_categorical
import rasterio.features
import numpy as np
import geopandas as gpd
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adagrad
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import precision_score, f1_score

# Function to load hyperspectral image data and metadata
def load_hyperspectral_image(path):
    with rasterio.open(path) as src:
        image_data = src.read()  # Read all bands of the hyperspectral image
        metadata = src.meta      # Read metadata including georeferencing
    return image_data, metadata

# Function to extract labeled training data from hyperspectral images using vector data
def extract_training_data(hyperspectral_image, metadata, geodataframe, label_column):
    training_data = []
    labels = []
    transform = metadata['transform']
    for index, feature in geodataframe.iterrows():
        geom = feature.geometry
        # Create a mask for the current geometry
        mask = rasterio.features.geometry_mask([geom], out_shape=(metadata['height'], metadata['width']),
                                               transform=transform, invert=True)
        # Find indices where the mask is True
        pixel_indices = np.argwhere(mask)
        for (y, x) in pixel_indices:
            pixel_value = hyperspectral_image[:, y, x]
            training_data.append(pixel_value)
            labels.append(feature[label_column])
    return np.array(training_data), np.array(labels)

# Function to create a convolutional neural network (CNN) model
def create_custom_cnn(input_shape, num_classes, filters1, kernel_size, filters2, filters3, pool_size):
    inputs = Input(shape=input_shape)
    x = Conv2D(filters1, kernel_size, activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size, padding='same')(x)
    x = Conv2D(filters2, kernel_size, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size, padding='same')(x)
    x = Conv2D(filters3, kernel_size, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size, padding='same')(x)
    x = Flatten()(x)
    x = Dense(filters3, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(filters2, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adagrad(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load hyperspectral data and geospatial data
raster_path = "/home/g98z895/projects/albedo/input/20240412_sodankyla_masked_2.tiff"
gdf_path = "/home/g98z895/projects/albedo/input/samples_sodankyla_polygons.gpkg"
hyperspectral_data, metadata = load_hyperspectral_image(raster_path)
gdf = gpd.read_file(gdf_path)

# Extract training data
training_data, labels = extract_training_data(hyperspectral_data, metadata, gdf, 'type')
if len(training_data) > 0:
    num_samples, num_bands = training_data.shape[0], training_data.shape[1]
    training_data = training_data.reshape(num_samples, 1, 1, num_bands)

    # Apply random over-sampling to address class imbalance
    ros = RandomOverSampler(random_state=0)
    training_data_reshaped = training_data.reshape(num_samples, -1)
    training_data_resampled, labels_resampled = ros.fit_resample(training_data_reshaped, labels)

    # Convert labels to one encoding
    num_classes = np.max(labels_resampled) + 1
    labels_resampled = to_categorical(labels_resampled, num_classes)

    # Reshape data back to its original dimensions
    training_data_resampled = training_data_resampled.reshape(-1, 1, 1, num_bands)

    # Iterate through different model configurations and train models
    filters_options_layer = [[16, 32, 64], [32, 64, 128], [64, 128, 256]]
    kernel_sizes_layer = [(3,3), (4,4), (5,5)]
    epochs_options = [25,50,75]
    pool_sizes = [(2,2)]
    results_df = pd.DataFrame(columns=['Kernel Size', 'Filters1', 'Filters2', 'Filters3', 'Pool Size', 'Epochs', 'Precision', 'F1 Score'])

    for filters in filters_options_layer:
        for kernel_size in kernel_sizes_layer:
            for pool_size in pool_sizes:
                for epochs in epochs_options:
                    model = create_custom_cnn((1, 1, num_bands), num_classes, filters[0], kernel_size, filters[1], filters[2], pool_size)
                    model.fit(training_data_resampled, labels_resampled, epochs=epochs, validation_split=0.2)
                    predictions = model.predict(training_data_resampled).argmax(axis=-1)
                    precision = precision_score(labels_resampled.argmax(axis=-1), predictions, average='macro')
                    f1 = f1_score(labels_resampled.argmax(axis=-1), predictions, average='macro')
                    results_df = pd.concat([results_df, pd.DataFrame([{'Kernel Size': kernel_size,
                                                                       'Filters1': filters[0],
                                                                       'Filters2': filters[1],
                                                                       'Filters3': filters[2],
                                                                       'Pool Size': pool_size,
                                                                       'Epochs': epochs,
                                                                       'Precision': precision,
                                                                       'F1 Score': f1}])], ignore_index=True)

    print(results_df.sort_values(by='F1 Score', ascending=False))
else:
    print("Insufficient data for training.")

# Save results to a file
results_df.to_csv('/home/g98z895/projects/albedo/output/paolleti.csv')

