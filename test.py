import os
import glob
import argparse
import matplotlib
import sys
import numpy as np
import cv2
from PIL import Image

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from tensorflow.keras.layers import Layer, InputSpec
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

model_file = 'nyu.h5'
# Argument Parser
image_path = sys.argv[1]

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(model_file, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(model_file))

# Input images
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
orig_width = image.shape[1]
orig_height = image.shape[0]
resized_image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
x = [np.clip(np.asarray(resized_image, dtype=float) / 255, 0, 1)]
input = np.stack(x, axis=0)
print('\nLoaded image of size {0}.'.format(input.shape[0:]))

# Compute results
output = predict(model, input)
output_resized = cv2.resize(output[0], (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
output_rescaled = cv2.normalize(output_resized, None, 0, 255, cv2.NORM_MINMAX)
new_pathname = image_path.replace('.', '_depthmap.')
cv2.imwrite(new_pathname, output_rescaled) 


fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Display first image on the first axis
axs[0].imshow(image)
axs[0].set_title('Original image')

# Display the second image (array) on the second axis
axs[1].imshow(output_rescaled, cmap='gray')  # Use cmap='gray' for grayscale images
axs[1].set_title('Depth Map')

# Hide axis ticks and labels
for ax in axs:
    ax.axis('off')

# Show the images side by side
plt.tight_layout()
plt.show()
