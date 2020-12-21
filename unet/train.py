import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import numpy as np

from model.unet import UNet


# download the dataset and get info
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

# see the possible keys we can access in the dataset dict.
# this contains the test and train splits.
print(dataset.keys())

# see information about the dataset
print(info)


# Preprocessing Utilities
def random_flip(input_image, input_mask):
    """does a random flip of the image and mask"""
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    return input_image, input_mask


def normalize(input_image, input_mask):
    """
    normalizes the input image pixel values to be from [0,1].
    subtracts 1 from the mask labels to have a range from [0,2]
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    """resizes, normalizes, and flips the training data"""
    input_image = tf.image.resize(datapoint["image"], (128, 128), method="nearest")
    input_mask = tf.image.resize(
        datapoint["segmentation_mask"], (128, 128), method="nearest"
    )
    input_image, input_mask = random_flip(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint):
    """resizes and normalizes the test data"""
    input_image = tf.image.resize(datapoint["image"], (128, 128), method="nearest")
    input_mask = tf.image.resize(
        datapoint["segmentation_mask"], (128, 128), method="nearest"
    )
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


# preprocess the train and test sets
train = dataset["train"].map(
    load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
test = dataset["test"].map(load_image_test)

BATCH_SIZE = 64
BUFFER_SIZE = 1000

# shuffle and group the train set into batches
train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

# do a prefetch to optimize processing
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# group the test set into batches
test_dataset = test.batch(BATCH_SIZE)

# class list of the mask pixels
class_names = ["pet", "background", "outline"]


def display_with_metrics(display_list, iou_list, dice_score_list):
    """displays a list of images/masks and overlays a list of IOU and Dice Scores"""

    metrics_by_id = [
        (idx, iou, dice_score)
        for idx, (iou, dice_score) in enumerate(zip(iou_list, dice_score_list))
        if iou > 0.0
    ]
    metrics_by_id.sort(key=lambda tup: tup[1], reverse=True)  # sorts in place

    display_string_list = [
        "{}: IOU: {} Dice Score: {}".format(class_names[idx], iou, dice_score)
        for idx, iou, dice_score in metrics_by_id
    ]
    display_string = "\n\n".join(display_string_list)

    display(
        display_list,
        ["Image", "Predicted Mask", "True Mask"],
        display_string=display_string,
    )


def display(display_list, titles=[], display_string=None):
    """displays a list of images/masks"""

    plt.figure(figsize=(15, 15))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
        if display_string and i == 1:
            plt.xlabel(display_string, fontsize=12)
        img_arr = tf.keras.preprocessing.image.array_to_img(display_list[i])
        plt.imshow(img_arr)

    plt.show()


def show_image_from_dataset(dataset):
    """displays the first image and its mask from a dataset"""

    for image, mask in dataset.take(1):
        sample_image, sample_mask = image, mask
    display([sample_image, sample_mask], titles=["Image", "True Mask"])


def plot_metrics(metric_name, title, ylim=5):
    """plots a given metric from the model history"""
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(model_history.history[metric_name], color="blue", label=metric_name)
    plt.plot(
        model_history.history["val_" + metric_name],
        color="green",
        label="val_" + metric_name,
    )


input_shape = (128, 128, 3, None)

model = UNet()
model.build(input_shape)
model.summary()
breakpoint()

# configure the optimizer, loss and metrics for training
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# configure the training parameters and train the model

TRAIN_LENGTH = info.splits["train"].num_examples
EPOCHS = 10
VAL_SUBSPLITS = 5
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VALIDATION_STEPS = info.splits["test"].num_examples // BATCH_SIZE // VAL_SUBSPLITS

# this will take around 20 minutes to run
model_history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    validation_data=test_dataset,
)

# Prediction Utilities


def get_test_image_and_annotation_arrays():
    """
    Unpacks the test dataset and returns the input images and segmentation masks
    """

    ds = test_dataset.unbatch()
    ds = ds.batch(info.splits["test"].num_examples)

    images = []
    y_true_segments = []

    for image, annotation in ds.take(1):
        y_true_segments = annotation.numpy()
        images = image.numpy()

    y_true_segments = y_true_segments[
        : (
            info.splits["test"].num_examples
            - (info.splits["test"].num_examples % BATCH_SIZE)
        )
    ]

    return (
        images[
            : (
                info.splits["test"].num_examples
                - (info.splits["test"].num_examples % BATCH_SIZE)
            )
        ],
        y_true_segments,
    )


def create_mask(pred_mask):
    """
    Creates the segmentation mask by getting the channel with the highest probability. Remember that we
    have 3 channels in the output of the UNet. For each pixel, the predicition will be the channel with the
    highest probability.
    """
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0].numpy()


def make_predictions(image, mask, num=1):
    """
    Feeds an image to a model and returns the predicted mask.
    """

    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
    pred_mask = model.predict(image)
    pred_mask = create_mask(pred_mask)

    return pred_mask


def class_wise_metrics(y_true, y_pred):
    class_wise_iou = []
    class_wise_dice_score = []

    smoothening_factor = 0.00001
    for i in range(3):

        intersection = np.sum((y_pred == i) * (y_true == i))
        y_true_area = np.sum((y_true == i))
        y_pred_area = np.sum((y_pred == i))
        combined_area = y_true_area + y_pred_area

        iou = (intersection + smoothening_factor) / (
            combined_area - intersection + smoothening_factor
        )
        class_wise_iou.append(iou)

        dice_score = 2 * (
            (intersection + smoothening_factor) / (combined_area + smoothening_factor)
        )
        class_wise_dice_score.append(dice_score)

    return class_wise_iou, class_wise_dice_score


# Setup the ground truth and predictions.

# get the ground truth from the test set
y_true_images, y_true_segments = get_test_image_and_annotation_arrays()

# feed the test set to th emodel to get the predicted masks
results = model.predict(
    test_dataset, steps=info.splits["test"].num_examples // BATCH_SIZE
)
results = np.argmax(results, axis=3)
results = results[..., tf.newaxis]

# compute the class wise metrics
cls_wise_iou, cls_wise_dice_score = class_wise_metrics(y_true_segments, results)

# show the IOU for each class
for idx, iou in enumerate(cls_wise_iou):
    spaces = " " * (10 - len(class_names[idx]) + 2)
    print("{}{}{} ".format(class_names[idx], spaces, iou))

# show the Dice Score for each class
for idx, dice_score in enumerate(cls_wise_dice_score):
    spaces = " " * (10 - len(class_names[idx]) + 2)
    print("{}{}{} ".format(class_names[idx], spaces, dice_score))
