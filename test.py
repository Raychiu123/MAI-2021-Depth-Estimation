import tensorflow as tf
import numpy as np
import imageio
import os

from model import UNet
from losses_tf import *
from logger import LoggingSingleton
from datetime import datetime

LoggingSingleton(datetime.now().strftime("%Y%m%d"), "ExecuteLog")
log = LoggingSingleton.get_instance()

np.random.seed(42)

IMAGE_WIDTH, IMAGE_HEIGHT, DEPTH_CHANNELS = 640, 480, 1

# Modify the model training parameters below:

BATCH_SIZE = 1
LEARNING_RATE = 2e-5
NUM_TRAIN_ITERATIONS = 50000
EVAL_STEP = 100

VAL_DIR = "../MAI2021_depth_valid_rgb/"

NUM_VAL_IMAGES = len(os.listdir(VAL_DIR))
NUM_VAL_BATCHES = NUM_VAL_IMAGES // BATCH_SIZE

def load_valid_data(dir_name):

    image_list = os.listdir(dir_name)
    dataset_size = len(image_list)

    image_ids = np.random.choice(np.arange(0, dataset_size), dataset_size, replace=False)

    rgb_data = np.zeros((dataset_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    image_name = []
    
    for i, img_id in enumerate(image_ids):
        I_rgb = imageio.imread(dir_name + image_list[img_id])
        rgb_data[i] = np.reshape(I_rgb, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        image_name.append(image_list[img_id])

    return rgb_data, image_name

#train_data, train_targets, val_data, val_targets = load_data(TRAIN_DIR, NUM_TRAIN_IMAGES)
#%%

with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as sess:

    # Placeholders for training data

    input_ = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    input_norm = input_ / 255.0         # mapping to [0, 1] interval

    # Get the predicted depth distance (in meters) and map it to millimeters

    predictions_raw = UNet(input_norm) * 1000

    # Clip the obtained values to uint16. The lower bound is set to 1mm to avoid problems when computing logarithms

    predictions = tf.clip_by_value(predictions_raw, 1.0, 65535.0)
    final_outputs = tf.cast(tf.clip_by_value(predictions_raw, 0.0, 65535.0), tf.uint16)

    print("Initializing variables")
    log.info("Initializing variables")

    sess.run(tf.compat.v1.global_variables_initializer())
    model_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith("model")]
    saver = tf.compat.v1.train.Saver(var_list=model_vars, max_to_keep=100)
    if os.path.isfile("./models/unet.ckpt.index"):
        print("ckpt exists")
        saver.restore(sess, "./models/unet.ckpt")
    else:
        print("no ckpt")

    # Export your model to the TFLite format
    converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [input_], [final_outputs])
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops=True
    converter.experimental_new_converter =True
    tflite_model = converter.convert()
    open("./models/model.tflite", "wb").write(tflite_model)
    
    print("Loading validation data...")
    log.info("Loading validation data...")
    test_data, ids = load_valid_data(VAL_DIR)
    print("Training/Validation data was loaded\n")
    log.info("Training/Validation data was loaded\n")

    for j in range(NUM_VAL_BATCHES):

        be = j * BATCH_SIZE
        en = (j + 1) * BATCH_SIZE

        input_test = test_data[be:en]
        imageID = ids[be:en]
        # Save visual results for several test images
        visual_results = sess.run(final_outputs, feed_dict={input_: input_test})

        i = 0
        for image in visual_results:
            predicted_image = np.asarray(np.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, DEPTH_CHANNELS]), dtype=np.uint16)
            imageio.imsave("results/" + imageID[i], predicted_image)
