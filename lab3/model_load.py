from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == "__main__":

    s = "flowers4.h5"
    
    model = load_model(s)

    
    test_datagen = ImageDataGenerator(rescale=1. / 255) 
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = val_datagen.flow_from_directory(
    'flowers_split/validation',
    target_size=(150, 150),
    batch_size=5,
    class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        'flowers_split/test',
        target_size=(150, 150),
        batch_size=5,
        class_mode='categorical')
    print(s)
    test_loss, test_acc = model.evaluate_generator(test_generator, steps=100)
    print('test acc:', test_acc)
    print('test loss:', test_loss)
    Y_pred = model.predict_generator(validation_generator, 173)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, y_pred))

    
