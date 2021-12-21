import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

img = image.load_img("/Users/matt/Downloads/project/neuralnetwork/train/banana/Banana_0.jpg")
plt.imshow(img)
plt.savefig('test.png')
cv2.imread("/Users/matt/Downloads/project/neuralnetwork/train/banana/Banana_0.jpg")
train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)

train_dataset = train.flow_from_directory('/Users/matt/Downloads/project/neuralnetwork/train/', target_size = (200,200), batch_size = 16, class_mode = 'categorical')
validation_dataset = validation.flow_from_directory('/Users/matt/Downloads/project/neuralnetwork/validation/', target_size = (200,200), batch_size = 16, class_mode = 'categorical')
# print(train_dataset.classes)

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    ##
                                    tf.keras.layers.Flatten(),
                                    ##
                                    tf.keras.layers.Dense(512, activation = 'relu'),
                                    ##
                                    tf.keras.layers.Dense(1, activation = 'softmax')                                    

]
)

model.compile(  loss = 'binary_crossentropy',
                optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
                metrics = ['accuracy'])

model_fit = model.fit(  train_dataset,
                        steps_per_epoch = 3,
                        epochs = 10,
                        validation_data = validation_dataset)

# serialize to JSON
json_file = model.to_json()
with open('/Users/matt/Downloads/project/neuralnetwork/model.json', "w") as file:
   file.write(json_file)
# serialize weights to HDF5
model.save_weights('model.hdf5')



dir_path = '/Users/matt/Downloads/project/neuralnetwork/tester/'

for i in os.listdir(dir_path):
    img = image.load_img(dir_path + i, target_size = (200,200))
    plt.imshow(img)
    #print("here-1")
    #plt.show()


    #print("here0")
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis = 0)
    images = np.vstack([X])
    #print("here1")
    val = model.predict(images)

    #print("here2")

    if val == 0:
        print("Banana")
    elif val == 1:
        print("Bread")
    elif val == 2:
        print("Eggs")
    elif val == 3:
        print("Milk")
    elif val == 4:
        print("Potato")
    elif val == 5:
        print("Spinach")
    elif val == 7:
        print("Tomato")
    else:
        print("Unidentified")

