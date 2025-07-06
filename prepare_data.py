import tensorflow as tf
import tensorflow_datasets as tfds

IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0  # Normalize to [0,1]
    return image, label

def prepare_dataset(dataset, shuffle=False):
    dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

# تحميل بيانات food101
dataset, info = tfds.load('food101', with_info=True, as_supervised=True)

train_ds = prepare_dataset(dataset['train'], shuffle=True)
val_ds = prepare_dataset(dataset['validation'])

print(f"Number of classes: {info.features['label'].num_classes}")
print(f"Class names: {info.features['label'].names}")
