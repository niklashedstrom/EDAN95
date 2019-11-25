from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def train():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
    input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc'])


    train_datagen = ImageDataGenerator(rescale=1. / 255,
    rotation_range=120,
    width_shift_range=0.5,
    height_shift_range=0.5,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = val_datagen.flow_from_directory(
    'flowers_split/validation',
    target_size=(150, 150),
    batch_size=5,
    class_mode='categorical')

    train_generator = train_datagen.flow_from_directory(
    'flowers_split/train',
    target_size=(150, 150),
    batch_size=15,
    class_mode='categorical')

    history = model.fit_generator(
    train_generator,
    steps_per_epoch=173,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

        

    model.save('flowers4.h5')


    test_generator = test_datagen.flow_from_directory(
        'flowers_split/test',
        target_size=(150, 150),
        batch_size=5,
        class_mode='categorical')

    test_loss, test_acc = model.evaluate_generator(test_generator, steps=100)
    Y_pred = model.predict_generator(validation_generator, 173)
    y_pred = np.argmax(Y_pred, axis=1)
    print('test acc:', test_acc)
    print('test loss:', test_loss)
    print('Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, y_pred))
    
    #model = load_model("flowers.h5")

    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))




if __name__ == "__main__": 
    train()