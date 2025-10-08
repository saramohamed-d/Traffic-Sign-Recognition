import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


model = tf.keras.models.load_model("best_model.keras")


classnames = [
    str(i) for i in range(43)
]

def preprocess_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [64,64])
    img = img/255.0
    img = tf.expand_dims(img, axis=0)
    return img

img_path = "./gtsrb/Test/00007.png"
img = preprocess_img(img_path)



pred = model.predict(img)
pred_class = np.argmax(pred, axis=1)[0]
confidence = np.max(pred)

print(f"Predicted class: {classnames[pred_class]} (Confidence: {confidence:.2f})")


plt.imshow(tf.Keras.utils.load_img(img_path))
plt.title(f"Predicted: {classnames[pred_class]}")
plt.axis("off")
plt.show()


