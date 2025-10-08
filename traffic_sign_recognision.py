import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

train = "./gtsrb/Train"
test = "./gtsrb/Test/"

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train,
    image_size=(64,64),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    train,
    image_size=(64,64),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=123
)

classnames = train_dataset.class_names
# print("Number of classes:", len(classnames))
# print("classes:", classnames)

normalization = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization(x), y))

train_dataset = train_dataset.cache().shuffle(1000).prefetch( buffer_size = tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)



test_csv= "./gtsrb/Test.csv"

with open(test_csv, "r") as f:
    lines = f.readlines()

header = lines[0].strip().split(",")

data = [line.strip().split(",")for line in lines[1:]]

test_df = pd.DataFrame(data, columns=header)

# print("Columns in CSV:", test_df.columns)

file_paths = [os.path.join("./gtsrb", img) for img in test_df["Path"]]
labels= test_df["ClassId"].astype(int).values



def load_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [64,64])
    img = img/255.0
    return img, label



test_dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
test_dataset = test_dataset.map(load_img).batch(32)





data_augmentation = tf.keras.Sequential([
     tf.keras.layers.RandomFlip("horizontal_and_vertical"),
     tf.keras.layers.RandomRotation(0.1),
     tf.keras.layers.RandomZoom(0.1),
     tf.keras.layers.RandomTranslation(0.1, 0.1),
]
)


model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same', input_shape=(64,64,3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),


    tf.keras.layers.Conv2D(256, (3,3), activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(classnames), activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

history = model.fit(
    train_dataset,
    validation_data = val_dataset,
    epochs = 20,
    callbacks=[checkpoint]
)

