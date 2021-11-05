from keras.initializers import glorot_uniform
def resnet_101(input_shape=(96,96,3),classes=10):
    x_input=Input(input_shape)
    x=ZeroPadding2D((3,3))(x_input)
    x=Conv2D(64,(7,7),strides=(2,2))(x)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)
    x=MaxPooling2D((3,3),strides=(2,2))(x)
    
    
    x=convolutional_block(x,f=3,filters=[64,64,256],s=1)
    x=identity_block(x,f=3,filters=[64,64,256])
    x=identity_block(x,f=3,filters=[64,64,256])
    
    
    x=convolutional_block(x,f=3,filters=[128,128,512])
    x=identity_block(x,f=3,filters=[128,128,512])
    x=identity_block(x,f=3,filters=[128,128,512])
    x=identity_block(x,f=3,filters=[128,128,512])
    
    
    x=convolutional_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    x=identity_block(x,f=3,filters=[256,256,1024])
    
    x=AveragePooling2D((2,2))(x)
#     x_fc=MaxPooling2D((2,2))(x)
    x=Flatten()(x)
    x=Dense(10,activation='softmax',
            kernel_initializer=glorot_uniform(seed=0))(x)
    
    model=Model(inputs=x_input,outputs=x)
    return model
