import tensorflow_datasets as tfds

# تحميل بيانات food101
dataset, info = tfds.load('food101', with_info=True, as_supervised=True)

print(info)  # معلومات عن البيانات

train_ds = dataset['train']
test_ds = dataset['validation']

# اختبار عرض مثال صورة وعلامتها (صنفها)
for image, label in train_ds.take(1):
    print("Image shape:", image.shape)
    print("Label:", label.numpy())
