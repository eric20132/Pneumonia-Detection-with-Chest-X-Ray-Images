#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import os, shutil
import numpy as np
#data path
base_dir = r"/home/helenawang/chest_xray"

train_original_dir = os.path.join(base_dir, "train")

validation_original_dir = os.path.join(base_dir, "val")

test_dir = os.path.join(base_dir, "test")

train_original_normal_dir = os.path.join(train_original_dir, "NORMAL")

train_original_pneu_dir = os.path.join(train_original_dir, "PNEUMONIA")

validation_original_normal_dir = os.path.join(validation_original_dir, "NORMAL")

validation_original_pneu_dir = os.path.join(validation_original_dir, "PNEUMONIA")

test_normal_dir = os.path.join(test_dir, "NORMAL")

test_pneu_dir = os.path.join(test_dir, "PNEUMONIA")

train_new_dir = os.path.join(base_dir, "train_new")
if not os.path.isdir(train_new_dir): os.mkdir(train_new_dir)

val_new_dir = os.path.join(base_dir, "val_new")
if not os.path.isdir(val_new_dir): os.mkdir(val_new_dir)

train_normal_dir = os.path.join(train_new_dir, "NORMAL")
if not os.path.isdir(train_normal_dir): os.mkdir(train_normal_dir)

train_pneu_dir = os.path.join(train_new_dir, "PNEUMONIA")
if not os.path.isdir(train_pneu_dir): os.mkdir(train_pneu_dir)

validation_normal_dir = os.path.join(val_new_dir, "NORMAL")
if not os.path.isdir(validation_normal_dir): os.mkdir(validation_normal_dir)

validation_pneu_dir = os.path.join(val_new_dir, "PNEUMONIA")
if not os.path.isdir(validation_pneu_dir): os.mkdir(validation_pneu_dir)

#process data
fnames = os.listdir(train_original_normal_dir)
np.random.shuffle(fnames)
length = len(fnames)
for i in range(length):
    if i < length*3//4:
        src = os.path.join(train_original_normal_dir, fnames[i])
        dst = os.path.join(train_normal_dir, fnames[i])
        shutil.copyfile(src, dst)
    else:
        src = os.path.join(train_original_normal_dir, fnames[i])
        dst = os.path.join(validation_normal_dir, fnames[i])
        shutil.copyfile(src, dst)

fnames = os.listdir(validation_original_normal_dir)
for fname in fnames:
    src = os.path.join(validation_original_normal_dir, fname)
    dst = os.path.join(validation_normal_dir, fname)
    shutil.copyfile(src, dst)

fnames = os.listdir(train_original_pneu_dir)
np.random.shuffle(fnames)
length = len(fnames)
for i in range(length):
    if i < length*3//4:
        src = os.path.join(train_original_pneu_dir, fnames[i])
        dst = os.path.join(train_pneu_dir, fnames[i])
        shutil.copyfile(src, dst)
    else:
        src = os.path.join(train_original_pneu_dir, fnames[i])
        dst = os.path.join(validation_pneu_dir, fnames[i])
        shutil.copyfile(src, dst)
        
fnames = os.listdir(validation_original_pneu_dir)
for fname in fnames:
    src = os.path.join(validation_original_pneu_dir, fname)
    dst = os.path.join(validation_pneu_dir, fname)
    shutil.copyfile(src, dst)

#build a CNN
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = "relu", input_shape = (150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = "relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation = "relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation = "relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))

from keras import optimizers
model.compile(loss = "binary_crossentropy",
              optimizer = optimizers.RMSprop(lr = 1e-4),
              metrics = ["acc"])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_new_dir,
    target_size = (150,150),
    batch_size = 39,
    class_mode = "binary")

validation_generator = test_datagen.flow_from_directory(
    val_new_dir,
    target_size = (150,150),
    batch_size = 13,
    class_mode = "binary")

history = model.fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 20,
    validation_data = validation_generator,
    validation_steps = 50)

#save model
model.save("chest_xray_model.h5")

#plot train and validation loss and accuracy
import matplotlib.pyplot as plt

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, "bo", label = "Training acc")
plt.plot(epochs, val_acc, "b", label = "Validation acc")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, loss, "bo", label = "Training loss")
plt.plot(epochs, val_loss, "b", label = "Validation loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.show()


#testset evaluate
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (150,150),
    batch_size = 10,
    class_mode = "binary")

results = model.evaluate(test_generator)
print("Test Loss:", results[0])
print("Test Accuracy:", results[1])