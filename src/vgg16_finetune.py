'''
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
#encoding=utf-8
import numpy as np
from keras.optimizers import SGD,Adagrad,RMSprop
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import save_model
from keras.models import load_model
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import TensorBoard,EarlyStopping,CSVLogger, ModelCheckpoint
from keras import applications
from keras.utils import plot_model
import sys
import os
from  vgg16 import VGG16
from  myConv import myConv
import argparse


##############################
# Configuration settings
##############################
def mkdir(path):

    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        print(path+' 创建成功')
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False


def preprocess_input_vgg(x):
    """Wrapper around keras.applications.vgg16.preprocess_input()
    to make it compatible for use with keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument.
    
    Parameters
    ----------
    x : a numpy 3darray (a single image to be preprocessed)
    
    Note we cannot pass keras.applications.vgg16.preprocess_input()
    directly to to keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument because the former expects a
    4D tensor whereas the latter expects a 3D tensor. Hence the
    existence of this wrapper.
    
    Returns a numpy 3darray (the preprocessed image).
    
    """
    from keras.applications.imagenet_utils import preprocess_input
    X = np.expand_dims(x, axis=0) # 3D 2 4D
    X = preprocess_input(X) # only 4D tensor ,RGB2BGR ,center:103 116 123 
    return X[0]


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-mn','--model_name',action='store',type=str,
        default = 'default',help='you must give a model name')
parser.add_argument('-dp','--data_path',action='store',type=str,
        default='../data',help='train and val  file path')
parser.add_argument('-lr','--learning_rate',action='store',type=float,
        default=0.01,help='learning_rate')
#parser.add_argument('-mt','--momentum',action='store',type=float,
#        default=0.9,help='learning_rate')
parser.add_argument('-ne','--num_epochs',action='store',type=int,
        default=10,help='num_epochs')
parser.add_argument('-bs','--batch_size',action='store',type=int,
        default=128,help='batch size')
parser.add_argument('-nc','--num_classes',action='store',type=int,
        default=2,help='num classes')   # no use now
parser.add_argument('-tl','--train_layers',nargs='+',action='store',type=str,
        default=['logit','linear'],help='layers need to be trained.')
# TODO
parser.add_argument('-tn','--top_N',action='store',type=int,
        default=5,help='whether the targets are in the top K predictions.')
parser.add_argument('-um','--use_model',action='store',type=str,
        default='',help='use model to initial.')
# TODO
parser.add_argument('-spe','--steps_per_epoch',action='store',type=int,
        default=100,help='train: steps_pre_epoch.')
parser.add_argument('-vs','--validation_steps',action='store',type=int,
        default=50,help='test: validation_steps.')


args = parser.parse_args()
print("="*50)
print("[INFO] args:\r")
print(args)
print("="*50)

train_data_dir = args.data_path + '/train_v1' #!!!!!!
validation_data_dir = args.data_path + '/test'

model_name = args.model_name

epochs = args.num_epochs

batch_size = args.batch_size

train_layers = args.train_layers

learning_rate  = args.learning_rate

use_model = args.use_model

steps_per_epoch = args.steps_per_epoch 

validation_steps = args.validation_steps 

S_PATH = sys.path[0]

DATA_PATH = args.data_path

TENSORBOARD_PATH = DATA_PATH + '/Graph/{}'.format(model_name)
mkdir(os.path.dirname(TENSORBOARD_PATH))
mkdir(TENSORBOARD_PATH)

LOG_PATH = DATA_PATH + '/log/training_{}.csv'.format(model_name)
mkdir(os.path.dirname(LOG_PATH))

BEST_WEIGHT = DATA_PATH + "/bestWeights/weight_{}.h5".format(model_name)
mkdir(os.path.dirname(BEST_WEIGHT))

END_WEIGHT = DATA_PATH + '/endWeights/weight_{}.h5'.format(model_name)
mkdir(os.path.dirname(END_WEIGHT))

END_MODEL = DATA_PATH + '/endModel/model_{}.h5'.format(model_name)
mkdir(os.path.dirname(END_MODEL))


if use_model == '' :
    print("*" * 50)
    print('[INFO] init train mode')
    print("*" * 50)

    # vgg16 
    vgg16 = VGG16(weights='imagenet')
    
    # ** get vgg top layer then add a logit layer for classification **
    fc2 = vgg16.get_layer('fc2').output
    x = Dropout(0.5)(fc2) # add 0413
    prediction = Dense(output_dim=1, activation='sigmoid', name='logit')(x)

##   bad method :)
#    flatten = vgg16.get_layer('flatten').output
#    fc1 = Dense(256, activation='relu',name='fc1')(flatten)
#    dropout = Dropout(0.5)(fc1)
#    fc2 = Dense(256, activation='relu',name='fc2')(dropout)
#    dropout = Dropout(0.5)(fc2)
#    prediction = Dense(output_dim=1, activation='sigmoid', name='logit')(dropout)

    model = Model(input=vgg16.input, output=prediction)

elif use_model == 'linear':
    #svm classification
    print("*" * 50)
    print('[INFO] use {} train mode'.format(use_model))
    print("*" * 50)

    # vgg16 
    vgg16 = VGG16(weights='imagenet')
    
    # ** get vgg top layer then add a logit layer for classification **
    fc2 = vgg16.get_layer('fc2').output
    prediction = Dense(output_dim=1, activation='linear', name='linear',kernel_regularizer=regularizers.l2(0.01))(fc2)
    model = Model(input=vgg16.input, output=prediction)


elif use_model == 'myConv':
    print("*" * 50)
    print('[INFO] use {} train mode'.format(use_model))
    print("*" * 50)
    model = myConv()
#    print("load weight")
#    model.load_weights("../data/endWeights/weight_v7.h5")


else:
    print("*" * 50)
    print('[INFO] continue train mode')
    print("*" * 50)
    model = load_model(use_model)


##############################
# which layer will be trained 
##############################
if use_model != 'myConv':
    for layer in model.layers:
        #if layer.name in ['fc1', 'fc2', 'logit']:
        if layer.name in train_layers :
            layer.trainable = True
        else:
            layer.trainable = False

# model summary and structure pic.
model.summary()


# only can be used in py2
# plot_model(model, show_shapes=True, show_layer_names=True,to_file='model.png')


##############################
# compile
##############################


# SGD version

#rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-06)
#lrrate = 4.e-4
#decay = lrrate / epochs
#sgd = SGD(lr=lrrate, momentum=0.9, decay=decay, nesterov=False)
#model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

# adagrad version
# adagrad = Adagrad(lr=learning_rate, epsilon=1e-06)
# model.compile(optimizer=adagrad, loss='binary_crossentropy', metrics=['accuracy'])

# adadelta drop 0.5
model.compile(optimizer='adadelta',
                      loss='binary_crossentropy',
                                    metrics=['accuracy'])

# svm version
# model.compile(loss='hinge',optimizer='adadelta',metrics=['accuracy','binary_crossentropy'])



##############################
# data generation
##############################

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(directory=train_data_dir,
                                                    target_size=[224, 224],
                                                    batch_size=batch_size,
                                                    class_mode='binary')

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg,
                                    )

validation_generator = validation_datagen.flow_from_directory(directory=validation_data_dir,
                                                              target_size=[224, 224],
                                                              batch_size=batch_size,
                                                              class_mode='binary')



##############################
# Call Back
##############################

# tensor board
tbCallBack = TensorBoard(log_dir=TENSORBOARD_PATH, histogram_freq=1,  
                  write_graph=True, write_images=True)
#* tensorboard --logdir path_to_current_dir/Graph --port 8080 
print("tensorboard --logdir {} --port 8080".format(TENSORBOARD_PATH))


# earlystoping
# ES = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

# csv log
csvlog = CSVLogger(LOG_PATH,separator=',', append=True)

# saves the model weights after each epoch if the validation loss decreased
checkpointer = ModelCheckpoint(filepath=BEST_WEIGHT, verbose=1, save_best_only=True)

#################################
# fit
#################################

# begin to fit 
model.fit_generator(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    callbacks=[tbCallBack,csvlog,checkpointer]);

#################################
# model
#################################

model.save_weights(END_WEIGHT)
save_model(model,END_MODEL)
