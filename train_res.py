model=resnet_101(input_shape=(96,96,3),classes=10)
model.compile(optimizer='adam',loss='categorical_crossentropy',
             metrics=['accuracy'])
model.summary()
nb_train_samples = 5000
nb_validation_samples = 8000
nb_epoch = 50
nb_classes = 10

(x_train,y_train),(x_test,y_test)=load_data()
# x_train=x_train.astype('float32')
# x_test=x_test.astype('float32')
# x_train/=255
# x_test/=255
y_train=np_utils.to_categorical(y_train-1,nb_classes)
y_test=np_utils.to_categorical(y_test-1,nb_classes)

train_datagen=ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
train_datagen.fit(x_train)
train_generator=train_datagen.flow(x_train,y_train,batch_size=32)
test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow(x_test, y_test, batch_size=32)

model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples,
    epochs=nb_epoch,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples)
