import glob
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

imgs = glob.glob("./img/*.png")

width  = 125
height = 50

x=[]
y=[]

for img in imgs:
    filename = os.path.basename(img)
    label = filename.split("_")[0]
    im = np.array(Image.open(img).convert("L").resize((width,height)))
    im = im /255
    x.append(im)
    y.append(label)

x = np.array(x)
x = x.reshape(x.shape[0],width,height,1)

def onehot_labels(values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape((len(integer_encoded),1))
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded
y = onehot_labels(y)
train_x , test_x, train_y, test_y = train_test_split(x,y,test_size=0.25,random_state=2)

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(width,height,1)))
model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(3,activation="softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="Adam",
              metrics=["accuracy"])
model.fit(train_x,train_y,
          epochs=35,
          batch_size=64)

score_train = model.evaluate(train_x,train_y)
print("Accuracy:",score_train[1]*100)
score_test = model.evaluate(test_x,test_y)
print("Test accuracy:",score_test[1]*100)

open("model.json","w").write(model.to_json())
model.save_weights("trex_weight.h5")