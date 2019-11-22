from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt
from keras.models import load_model

if __name__ == "__main__":

    model = load_model("flowers2.h5")

    
    test_datagen = ImageDataGenerator(rescale=1. / 255) 
    #rotation_range=40,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True,
    #fill_mode='nearest')
    #)

    test_generator = test_datagen.flow_from_directory(
        'flowers_split/test',
        target_size=(150, 150),
        batch_size=5,
        class_mode='categorical')

    test_loss, test_acc = model.evaluate_generator(test_generator, steps=100)
    print('test acc:', test_acc)



