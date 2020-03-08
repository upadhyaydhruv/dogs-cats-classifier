import shutil, os
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import matplotlib.pyplot as plt

# orig_dir contains original files
orig_dir = '/content/drive/My Drive/train/'

# gen_dir will contain only disposable generated stuff, as it will be removed on
# each run
gen_dir = '/content/_gen/' 
shutil.rmtree(gen_dir, ignore_errors=True) #This recursively removes previous instances of all gen_dir files
os.makedirs(gen_dir) #This creates a new gen_dir directory

# sort the files
for thing in ["cat", "dog"]: #This partitions all of the files based on their label (cat, dog) and amount
  for (nums, animal) in [
    (range(800),        "train"), #First 500 of dogs and cats go into the training set
    (range(801,  1000), "val"), #Next 500 into validation set
    (range(1000, 1499), "test"), #Final 500 in the test set
  ]:
    #os.path.join creates directories under gen_dir in content called train_dir, test_dir, and val_dir
    #Within train_dir, there are directories created called train_dogs and train_cats (done for val and test as well)
    out = os.path.join(gen_dir, f"{animal}_dir", f"{animal}_{thing}s") 

    #makedirs makes the directories established in the above line
    os.makedirs(out)
    print(out) #Just for reference purposes
    for fn in [f"{thing}.{i}.jpg" for i in nums]:
      #Copies the files in the form cat.0.jpg (or dog.0.jpg) where the 0 is any number from the train file in My Drive to the directories established above
      shutil.copyfile(os.path.join(orig_dir, fn), os.path.join(out, fn)) 
      
#Adds the layers for the neural network
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape = (150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5));
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

train_dir = '/content/_gen/train_dir/'
val_dir = '/content/_gen/val_dir/'
test_dir = '/content/_gen/test_dir/'
train_datagen = ImageDataGenerator(rescale=1./255,
                                   #The factors below are used for data augmentation
                                   #Which essentially does a series of transformations 
                                   #on the images to create more data for the training
                                   #set to prevent overfitting
                                   rotation_range=40, 
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(150,150),
                                                    batch_size=32, 
                                                    class_mode='binary')
validation_generator = val_datagen.flow_from_directory(val_dir,
                                                        target_size=(150,150),
                                                        batch_size=32,
                                                        class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(150,150),
                                                  batch_size=32,
                                                  class_mode='binary')

history = model.fit_generator(
    train_generator, 
    steps_per_epoch=100, 
    epochs=23,
    validation_data = validation_generator,
    validation_steps=50)

model.save('cats_and_dogs_small_1.h5')

#A script that uses matplotlib to plot training vs validation loss and accuracy to effectively train the network

train_acc = history.history['acc']
val_acc = history.history['val_acc']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(train_acc)+1)

plt.plot(epochs, train_acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title("Training and Validation Accuracy - Model")
plt.legend()
plt.show()

plt.plot(epochs, train_loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss - Model')
plt.legend()
plt.show()
