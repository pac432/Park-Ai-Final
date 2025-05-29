# imports
# base: traffic
# adds: sklearn + extra tf
# matlplotlib if i want to return cool imgs
import os
import numpy as np
# read files
import cv2
# machine learning
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import KFold
# pictures :)
import matplotlib.pyplot as plt

# most ct scans are 512 x 512
IMG_HEIGHT = 512
IMG_WIDTH = 512

# loading scans & masks
# not quite sure i used the right data type
# 170th slice bc thats roughly where the heart is
def load_slice(path, slice_index=169, width=512, height=512, dtype=np.h.265):
    
    # using number of slices to find depth
    pixel_size = np.dtype(dtype).itemsize
    slice_size = width * height * pixel_size
    file_size = os.path.getsize(path)
    depth = file_size / slice_size

    # requested slice exists?
    if not (0 <= slice_index < depth):
        raise ValueError(f"Requested slice does not exist")

    # Open and seek to the slice
    with open(path, 'rb') as f:
        offset = slice_index * slice_size * np.dtype(dtype).itemsize
        f.seek(offset)
        # returns as 1d slice
        # line from gemini
        raw = np.frombuffer(f.read(slice_size * np.dtype(dtype).itemsize), dtype=dtype)

    # return slice to 2d
    img = raw.reshape((height, width))

    # change data type to float to be ai friendly and make the img values 0-1
    img = img.astype('float32') / np.max(img)
    return img

# PATHS = USEFUL BOOKMARK <================================================================>
scan_path = "/Users/andrew/downloads/finalset"
mask_path = "/Users/andrew/downloads/seg-LUNGS-Luna16"
test_path = "/Users/andrew/downloads/test set"
test_mask_path = "/Users/andrew/downloads/test_mask_set"
# end bookmark <==========================================================================>

# did you parse your data correctly?
def data_check(scan_dir, mask_dir, test_dir, test_mask_dir, width, height):
    scans = []
    masks = []
    test = []

    # access folders
    scan_files = set(os.listdir(scan_dir))
    mask_files = set(os.listdir(mask_dir))
    test_files = set(os.listdir(test_dir))
    test_mask_files = set(os.listdir(test_mask_dir))

    # check for misaligned folder lens
    # trying to flag incorrectly pared mask folder
    # this is assuming no exess scans were downloaded
    # Todo: READ README TO CHECK FOR ^
    scan_count = len(mask_files - scan_files)
    if scan_count != 0:
        extra_masks = mask_files - scan_files
        raise ValueError("Extra Masks:" {list(extra_masks)})

    # compile files
    for filename in scan_files:
        scan_path = os.path.join(scan_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        test_path = os.path.join(test_dir, filename)
        mask_path = os.path.join(test_mask_dir, filename)
        # 170th slice bc thats roughly where heart is
        scans.append(load_slice(scan_path,  width, height, slice_index=169))
        masks.append(load_slice(mask_path, width, height, slice_index=169))
        test.append(load_slice(test_path, width, height, slice_index=169))
        test_mask.append(load_slice(test_mask_path,width, height, slice_index=169))
    # set => array
    return np.array(scans), np.array(masks), np.array(test), np.array(test_mask)


# Load data + file check
images, masks, test, test_mask = data_check(scan_path, mask_path, test_path, test_mask_path, 512, 512)

# Build the U-net
# U shaped detail curve, encoders pool down resolution, decoders concatenate up resolution
# boilerplate from gemini, overhauled/simplified by me
# collapsed manually adding entire blocks to single functions

# encoder
def encoder(inputs, num_filters):
    x = bttlnk(inputs, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

# decoder
def decoder(inputs, skip_features, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    x = layers.concatenate([x, skip_features])
    x = bttlnk(x, num_filters)
    return x

# bottleneck - anti overfitting here (dropout)
def bttlnk(inputs, num_filters, dropout_rate=0.20):
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(inputs)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    return x

# compiling the network
# recursive design
# each layer doubles filter num
# play around with it if time?
# recursively build block by block to change model depth with one var tweak?
# DONT recursivly build the net - my implementations keep nuking previous layers
def unet(input_shape):
    # splits 2d (or 3 by now) array into 1d parameters
    inputs = layers.Input(input_shape)
    
    # pooling
    s1, p1 = encoder(inputs, 64)
    s2, p2 = encoder(p1, 128)
    s3, p3 = encoder(p2, 256)
    s4, p4 = encoder(p3, 512)
    
    # bottleneck
    b1 = bttlnk(p4, 1024)

    # concenating
    d1 = decoder(b1, s4, 512)
    d2 = decoder(d1, s3, 256)
    d3 = decoder(d2, s2, 128)
    d4 = decoder(d3, s1, 64)
    outputs = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(d4)
    return models.Model(inputs, outputs)

# training - k fold cross validation + test
# not adding epochs. wouldnt be able to test it on my computer anyways
def train(images, masks, epochs, batch_size, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    # 1 withhold
    fold = 1
    fold_results = []

    # looping each training set in a cycle
    for train_val, true_val in kf.split(images):
        # splitting into train anc validation
        X_train, X_val = images[train_val], images[true_val]
        y_train, y_val = masks[train_val], masks[true_val]

        # assuming Luna16
        model = unet((512, 512))
        # compiling
        # using adam optimiser bc it goes easy on my computer (and name lol) 
        # using binary crossentropy as loss function for hard borders
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model on this fold's training data
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)

        # evalutate
        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"Fold {fold} — Val Loss: {loss:.4f}, Val Acc: {acc:.4f}")
        # storing in array to break for loop
        fold_results.append({'fold': fold, 'loss': loss, 'accuracy': acc})
        fold += 1

# training results
    # averaging results from each
    avg_loss = np.mean([r['loss'] for r in fold_results])
    avg_acc = np.mean([r['accuracy'] for r in fold_results])
    
    # results to 3 sigfigs for now
    print(f"\nAverage over {k} folds:")
    print(f"\nTraining Loss: {avg_loss:.3f}")
    print(f"\n Accuracy: {avg_acc:.3f}")

    # testing results
    test_loss, test_acc = model.evaluate(test, test_mask, verbose=0)
    print(f"\nTesting Averages:")
    print(f"\nTesting Loss: {test_loss:.3f}")
    print(f"\nTesting Accuracy: {test_acc:.3f}")

#run training +test
epochs = 10
batch_size = 16
train(images, masks, test, test_mask, epochs, batch_size)


