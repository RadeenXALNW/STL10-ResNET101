# x is input, y=F(x)
# identity block simply means input should be equal to output. 
#  y = x + F(x)   the layers in a traditional network are learning the true output H(x)
# F(x) = y - x   the layers in a residual network are learning the residual F(x)
# Hence, the name: Residual Block.


def identity_block(x,f,filters):
    """
   
    Arguments:
    X -- input of shape (m, height, width, channel)
    f -- shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    F1,F2,F3=filters
    x_skip=x
    
    #first layer
    x=Conv2D(filters=F1,kernel_size=(1,1),strides=(1,1),padding='valid')(x)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)
    
    #second layer
    x=Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),padding='same')(x)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)
    
    #third layer
    x=Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding='valid')(x)
    x=BatchNormalization(axis=3)(x)
    
    #finally we will add the skip value to the last convolve result with a relu activation
    x=Add()([x,x_skip])
    x=Activation('relu')(x)
    
    return x
