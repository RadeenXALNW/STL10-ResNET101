# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(96, 96))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 96, 96, 3)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

 
# load an image and predict the class
def run_example():
    # load the image
    img = load_image('D:/deep learning research paper/cnn from scratch/stl10/unlabeled_images/unlabeled_image_png_157.png')
    # load model
    model = load_model('my_model',compile=False)
    # predict the class
    result = model.predict(img)
    x=np.argmax(result,axis=1)
    print([int(x)])
 
# entry point, run the example
load_image('D:/deep learning research paper/cnn from scratch/stl10/unlabeled_images/unlabeled_image_png_157.png')
run_example()
