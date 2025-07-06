import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

# الأصناف اللي بنشتغل عليها
selected_classes = ['pizza', 'sushi', 'falafel', 'waffles', 'ice_cream']
IMG_SIZE = 224

# تحميل البيانات
dataset, info = tfds.load('food101', with_info=True, as_supervised=True)
class_names = info.features['label'].names

# نجهز قاموس لتحويل اسم الصنف إلى رقمه
class_name_to_index = {name: idx for idx, name in enumerate(class_names)}
selected_class_indices = [class_name_to_index[name] for name in selected_classes]

# تصفية الداتا
def filter_classes(image, label):
    return tf.reduce_any(tf.equal(label, selected_class_indices))

def remap_labels(image, label):
    new_label = tf.argmax(tf.cast(tf.equal(label, selected_class_indices), tf.int32))
    return image, new_label

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

# تجهيز البيانات
train_ds = dataset['train'] \
    .filter(filter_classes) \
    .map(remap_labels) \
    .map(preprocess) \
    .batch(32) \
    .prefetch(tf.data.AUTOTUNE)

val_ds = dataset['validation'] \
    .filter(filter_classes) \
    .map(remap_labels) \
    .map(preprocess) \
    .batch(32) \
    .prefetch(tf.data.AUTOTUNE)

# بناء النموذج
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(len(selected_classes), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# تدريب النموذج
model.fit(train_ds, validation_data=val_ds, epochs=5)

# حفظ النموذج
model.save('food_cnn_mobilenetv2_selected.h5')
print("✅ تم تدريب النموذج على الأصناف المختارة بنجاح وحفظه!")
