import shutil, os

# orig_dir contains original files
orig_dir = '/content/drive/My Drive/train/'

# gen_dir will contain only disposable generated stuff, as it will be removed on
# each run
gen_dir = '/content/drive/My Drive/_gen/' 
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

from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import matplotlib.pyplot as plt
from keras.applications import MobileNet
from keras.optimizers import Adam

#Adds the layers for the neural network

def create_model():
  model = models.Sequential()
  model.add(MobileNet(weights='imagenet', include_top=False))
  model.add(layers.GlobalAveragePooling2D())
  model.add(layers.Dropout(rate=0.5))
  model.add(layers.Dense(1024, activation='relu'))
  model.add(layers.Dense(1024, activation='relu'))
  model.add(layers.Dense(512, activation='relu'))
  model.add(layers.Dense(1, activation='sigmoid'))
  model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

  for layer in model.layers:
    layer.trainable = True
  return model

train_dir = '/content/drive/My Drive/_gen/train_dir/'
val_dir = '/content/drive/My Drive/_gen/val_dir/'
test_dir = '/content/drive/My Drive/_gen/test_dir/'
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
                                                    target_size=(224,224),
                                                    batch_size=32, 
                                                    class_mode='binary',
                                                    shuffle=True)
validation_generator = val_datagen.flow_from_directory(val_dir,
                                                        target_size=(224,224),
                                                        batch_size=32,
                                                        class_mode='binary',
                                                       shuffle=True)

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(224,224),
                                                  batch_size=32,
                                                  class_mode='binary',
                                                  shuffle=True)

model = create_model()

print(model.summary())

history = model.fit_generator(
    train_generator, 
    steps_per_epoch=100, 
    epochs=7,
    validation_data = validation_generator,
    validation_steps=50)

model.save('cats_and_dogs_small_1.h5')

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
