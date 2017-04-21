'''
@clz
model feedforward prediction
'''

from PIL import Image
from keras.models import save_model
from keras.models import load_model
from keras.models import Model
from sklearn.svm import SVC
import numpy as np
import sys
import glob
import getImage4Predict
import pickle
import os

model_path = sys.argv[1]
pic_path = sys.argv[2]
out_fix_name = sys.argv[3]

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

def save(files,out,path='../data/output/out{}.csv'.format(out_fix_name)):
    '''
    保存结果
    '''
    with open(path,'w') as inf:
        for filename,out in zip(files,out):
            print(filename)
            print(out)
            name = os.path.basename(filename).split('.')[0] # id get!
            inf.write("{},{}\n".format(name,out[1])) # get label prob !!

def data2svm(files,out):
    '''
    转换label 0 1
    '''
    check = lambda x: 1 if 'dog' in x else 0
    labels = [check(i) for i in files]
    return labels,out
    

def svc(traindata,trainlabel,testdata,testlabel):
    '''
    svm
    '''
    print("Start training SVM...")
    print(traindata.shape)
    print(trainlabel.shape)
    svcClf = SVC(C=1.0,kernel="rbf",cache_size=40000,verbose=True,probability=True,max_iter=-1)
    #svcClf = SVC(C=1.0,kernel="sigmoid",cache_size=20000,verbose=True,probability=True)
    svcClf.fit(traindata,trainlabel)

    print("Start testing SVM cv...")
    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-svm Accuracy:",accuracy)
    return svcClf

def getPic(path,fixLabel):
    '''
    获取这个文件夹的图片
    '''
    print("[INFO] BEGIN TO READ:{}>>>".format(path))
    files,ims = getImage4Predict.getPics(path)
    print("datashape:{}".format(ims.shape))
    labels = np.ones(len(ims)) * fixLabel
    return files,np.array(ims),np.array(labels)

def getModel(model_path,layyerOut='fc2'):
    '''
    get model
    '''
    # load model
    model = load_model(model_path)
    fc2Out = model.get_layer('fc2').output
    outModel = Model(input=model.input, output=fc2Out)
    outModel.summary()
    return outModel

def predict(model,ims):
    out = model.predict(ims)
    return out

if __name__=="__main__":

    # parse dara
    DATAPATH = '../data/'
    trainPath = DATAPATH + 'train_v1' #11111111111111111111111
    testPath = DATAPATH + 'test'

#    #train data
#    _,dogImg,dogLabel = getPic(trainPath+"/dogs/",1)
#    _,catImg,catLabel = getPic(trainPath+"/cats/",0)
#    print(dogLabel)
#    print(catLabel)
#    trainImg = np.concatenate((dogImg,catImg),axis=0)
#    trainLabel = np.concatenate((dogLabel,catLabel),axis=0)
#
#    #get test data
#    _,dogImg,dogLabel = getPic(testPath+"/dogs/",1)
#    _,catImg,catLabel = getPic(testPath+"/cats/",0)
#    testImg = np.concatenate((dogImg,catImg),axis=0)
#    testLabel = np.concatenate((dogLabel,catLabel),axis=0)
#    print("[INFO]\r\
#            trainFea:{}\r\
#            trainLable:{}\r\
#            testFea:{}\r\
#            testLabel:{}\r\
#            ".format(trainImg.shape,trainLabel.shape,
#                testImg.shape,testLabel.shape))
#
#   #load model
#    print("[INFO] begin to load model>>")
#    model = getModel(model_path,'fc2')
#    print("[INFO] begin to get NN feature>>>")
#    trainFeature = predict(model,trainImg)
#    testFeature = predict(model,testImg)
#    print(trainFeature.shape)
#    print(testFeature.shape)
#    print("[INFO] begin to save NN feature>>>")
#    np.save("../data/tmp/trainFea",trainFeature)
#    np.save("../data/tmp/testFea",testFeature)
#    np.save("../data/tmp/trainLabel",trainLabel)
#    np.save("../data/tmp/testLabel",testLabel)

#    # get  data
#    print("[INFO] begin to get true test feature>>>")
    files,Img,_ = getPic(pic_path,0)
#    print(Img)
#    model = getModel(model_path,'fc2')
#    ImgFeature = predict(model,Img)
#    np.save("../data/tmp/trueTestLabel",ImgFeature)
#
#    # svc
#    print("[INFO] begin to load NN feature>>>")
#    trainFeature = np.load("../data/tmp/trainFea.npy")
#    testFeature = np.load("../data/tmp/testFea.npy")
#    trainLabel = np.load("../data/tmp/trainLabel.npy")
#    testLabel = np.load("../data/tmp/testLabel.npy")
#    print(trainFeature.shape)
#
#    print("[INFO] begin to predict >>>")
#    svcClf = svc(trainFeature,trainLabel,testFeature,testLabel)
#
#    # save
#    svm_model="../data/model/svm_{}".format(out_fix_name)
#    with open(svm_model,'wb') as f:
#        pickle.dump(svcClf,f) #save
#
#    # predict test
#    print("Start predict SVM...")
#    with open(svm_model,'rb') as f:
#        svcClf = pickle.load(f) #save
#    ImgFeature = np.load("../data/tmp/trueTestLabel.npy")
#    pred_testlabel_v1 = svcClf.predict(ImgFeature)
#    print(pred_testlabel_v1)
#    pred_testlabel_v2 = svcClf.predict_proba(ImgFeature)
#    #pre_prob = [i[0] for i in pred_testlabel_v2]
#    #print(np.array(pre_prob))
#
#    print("Start save SVM...")
#    np.save("../data/tmp/predict_v1.npy",pred_testlabel_v1)
#    np.save("../data/tmp/predict_v2.npy",pred_testlabel_v2)

    pred_testlabel_v2 = np.load("../data/tmp/predict_v2.npy")
    save(files,pred_testlabel_v2)
#python vgg16_svm.py ../data/endModel/model_v12.h5 ../data/testPic/test svmtest
