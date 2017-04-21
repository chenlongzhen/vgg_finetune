'''
@clz
model feedforward prediction
'''

from PIL import Image
from keras.models import save_model
from keras.models import load_model
import numpy as np
import sys,os
import glob
import getImage4Predict

model_path = sys.argv[1]
pic_path = sys.argv[2]
out_fix_name = sys.argv[3]
# load model
model = load_model(model_path)

#
#def get_pic(path):
#    pathList = glob.glob(path+'/*')
#
#
#im = Image.open(pic_path).resize((224, 224), Image.ANTIALIAS)
#im = np.array(im).astype(np.float32) # 2array
#
## scale the image, according to the format used in training
#im[:,:,0] -= 103.939
#im[:,:,1] -= 116.779
#im[:,:,2] -= 123.68
#im = im.transpose((1,0,2))
#im = np.expand_dims(im, axis=0)
#print(im.shape)


def save(files,out,path='../data/output/out_{}.csv'.format(out_fix_name)):
    with open(path,'w') as inf:
        for filename,out in zip(files,out):
            
            name = os.path.basename(filename).split('.')[0] # id get!
            print("{},{}".format(name,out[0]))
            inf.write("{},{}\n".format(name,out[0]))

print("[INFO] BEGIN TO READ>>>")
files,ims = getImage4Predict.getPics(pic_path)
print(files)
print("datashape:{}".format(ims.shape))

# out
print("[INFO] BEGIN TO PREDICT>>>")
out = model.predict(ims)
print("files:{}".format(files))
print("inference:{}".format(out))

save(files,out)
#python vgg16_feedforward.py model/vgg_dog_cat_v1.h5 ./testPic/test/  v1
