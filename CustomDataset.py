DATA_DIR = 'D:/deep learning research paper/cnn from scratch/New folder/stl10_binary'
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels
    
def read_all_images(path_to_data):
    with open(path_to_data,'rb') as f:
# We force the data into 3x96x96 chunks, since the
# images are stored in "column-major order", meaning
# that "the first 96*96 values are the red channel,
# the next 96*96 are green, and the last are blue."
# The -1 is since the size of the pictures depends
# on the input file, and this way numpy determines
# the size on its own

        all_images=np.fromfile(f,dtype=np.uint8)
        images=np.reshape(all_images,(-1,3,96,96))
        images=np.transpose(images,(0,3,2,1))
        return images

def read_single_image(image_file):
    """
    CAREFUL! - this method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    image = np.transpose(image, (2, 1, 0))
    return image



def load_data():
    # download data if needed
    path = get_file(DATA_DIR, origin=DATA_URL, untar=True)

    # test to check if the whole dataset is read correctly
    # path to the binary train file with image data
    train_data_path = os.path.join(path, 'D:/deep learning research paper/cnn from scratch/New folder/stl10_binary/train_X.bin')

    # path to the binary train file with labels
    train_label_path = os.path.join(path, 'D:/deep learning research paper/cnn from scratch/New folder/stl10_binary/train_y.bin')

    # path to the binary test file with image data
    test_data_path = os.path.join(path, 'D:/deep learning research paper/cnn from scratch/New folder/stl10_binary/test_X.bin')

    # path to the binary test file with labels
    test_label_path = os.path.join(path, 'D:/deep learning research paper/cnn from scratch/New folder/stl10_binary/test_y.bin')

    x_train = read_all_images(train_data_path)
    print(x_train.shape)

    y_train = read_labels(train_label_path)
    print(y_train.shape)

    x_test = read_all_images(test_data_path)
    print(x_test.shape)

    y_test = read_labels(test_label_path)
    print(y_test.shape)

    return (x_train, y_train), (x_test, y_test)
