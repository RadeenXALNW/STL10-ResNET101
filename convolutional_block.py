def convolutional_block(x,f,filters,s=2):
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
    x=Conv2D(filters=F1,kernel_size=(1,1),strides=(s,s),padding='valid')(x)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)
    
    #second layer
    x=Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),padding='same')(x)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)
    
    #third layer
    x=Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding='valid')(x)
    x=BatchNormalization(axis=3)(x)
    
    
    #skip part
    
    x_skip=Conv2D(filters=F3,kernel_size=(1,1),strides=(s,s),padding='valid')(x_skip)
    x_skip=BatchNormalization(axis=3)(x_skip)
    
    # Final step: Add shortcut value here, and pass it through a RELU activation 
    x=Add()([x,x_skip])
    x=Activation('relu')(x)
    
    return x
