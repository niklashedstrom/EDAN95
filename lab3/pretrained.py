
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
    


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
    directory,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical')
    
    i=0
    for inputs_batch, labels_batch in generator:
        #print(inputs_batch)
        #print(labels_batch)
        #print("-" * 20)
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        print(features_batch)
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels
    
#train 2595



conv_base = VGG16(weights='imagenet',
include_top=False,
input_shape=(150, 150, 3))

batch_size = 15

datagen = ImageDataGenerator(rescale=1. / 255)

train_dir = "flowers_split/train"
validation_dir = 'flowers_split/validation'
test_dir = 'flowers_split/test'


train_features, train_labels = extract_features(train_dir, 2595)
validation_features, validation_labels = extract_features(validation_dir, 865)
test_features, test_labels = extract_features(test_dir, 865)
train_features = np.reshape(train_features, (2595, 4*4* 512))
validation_features = np.reshape(validation_features, (865, 4*4* 512))
test_features = np.reshape(test_features, (865, 4*4* 512))

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
loss='categorical_crossentropy',
metrics=['acc'])
history = model.fit(train_features, train_labels,
epochs=30,
batch_size=batch_size,
validation_data=(validation_features, validation_labels))

test_loss, test_acc = model.evaluate_generator(test_generator, steps=100)
print('test acc:', test_acc)