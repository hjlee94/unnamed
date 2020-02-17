from unnamed.classification.interface import DatasetInterface
from unnamed.network_architecture.transformation.autoencoder import AutoEncoder
from unnamed.preprocessing import DataPreprocessor
from PIL import Image
import numpy as np
import glob


def load_data(path):
    X = list()

    image_path_list = glob.glob(path)

    for image_path in image_path_list[:150]:
        img = Image.open(image_path)
        img = img.resize((128, 128))
        # print(img.format, img.size, img.mode)
        # img = np.array(img).reshape(3, 128, 128)
        img = np.array(img)
        img = np.moveaxis(img, -1, 0)
        img = img / 256.0
        X.append(img)

    X = np.array(X)
    print(X.shape)
    return X

def to_image(x, name):
    x = np.array(x * 256, dtype=np.uint8)
    x[x > 255] = 255
    print(x.shape)
    x = Image.fromarray(x)
    x.show()
    x.save('%s.jpg'%name, "JPEG")


X = load_data('D:/dogs-vs-cats/train//train//cat.*.jpg')

autoencoder = AutoEncoder('conv')
autoencoder.fit(X)

X_transformed = autoencoder.transform(X)
print(X_transformed.shape)

X_restorted = autoencoder.inverse_transform(X_transformed)
print(X_restorted.shape)

# random_img = np.random.random_sample(size=(6,30,30))
# random_img = random_img.reshape((1,6,30,30))
# random_img = autoencoder.inverse_transform(random_img)
# to_image(np.moveaxis(random_img[0], 0, -1), 'test')

junho = Image.open('junho.jpg')
junho = junho.resize((128, 128))
junho = np.array(junho, dtype=float)
junho = np.moveaxis(junho, -1, 0)
junho = junho / 256.0
junho = np.array(junho).reshape((1,3,128,128))

junho = autoencoder.transform(junho)
junho = autoencoder.inverse_transform(junho)

to_image(np.moveaxis(junho[0], 0, -1), 'juncat')

# for target_idx in range(100):
#     to_image(np.moveaxis(X[target_idx], 0, -1), str(target_idx)+"_org")
#     to_image(np.moveaxis(X_restorted[target_idx], 0, -1), str(target_idx)+"_generalized")
#     input("next?")

# dd = DatasetInterface('./resource/iris.csv', label_pos=-1, preprocess_method='scale')
# print(dd)
# X, y = dd.getXY()
#
# autoencoder = AutoEncoder('basic')
# autoencoder.fit(X)
#
# print(X)
#
# X_transformed = autoencoder.transform(X)
#
# print(X_transformed)
#
# X_inverse_transformed = autoencoder.inverse_transform(X_transformed)
#
# print(X_inverse_transformed)
