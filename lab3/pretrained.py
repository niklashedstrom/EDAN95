
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
from sklearn.metrics import confusion_matrix
    


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count, 5))
    generator = datagen.flow_from_directory(
    directory,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical')
    
    i=0
    for inputs_batch, labels_batch in generator:
        print('yaho + {}'.format(i))
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
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
    #rotation_range=40,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True,
    #fill_mode='nearest')

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
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
    loss='categorical_crossentropy',
    metrics=['acc'])

history = model.fit(train_features, train_labels,
    epochs=30,
    batch_size=batch_size,
    validation_data=(validation_features, validation_labels))

model.save("pretrained.h5")

results = model.evaluate(test_features, test_labels, batch_size=batch_size)
print('test loss, test acc:', results)

Y_pred = model.predict(validation_features, batch_size=batch_size)
y_pred = np.argmax(Y_pred, axis=1)
y_test = np.argmax(validation_labels, axis=1)
#print('test acc:', test_acc)
# print('test loss:', test_loss)
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
