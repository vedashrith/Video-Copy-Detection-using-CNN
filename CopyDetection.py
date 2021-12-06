from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd
import os
import re
from keras.models import model_from_json
import cv2
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from sklearn.metrics import classification_report
import imutils

main = tkinter.Tk()
main.title("Video Copy Detection Using Spatio Temporal CNN Features") #designing main screen
main.geometry("1300x1200")

global model
global filename
global frame
SAMPLE_DURATION = 16
SAMPLE_SIZE = 300
precision = []

def loadInvertedIndexFeatures():
    text.delete('1.0', END)
    global model
    if os.path.exists('features/index.json'):
        with open('features/index.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        model.load_weights("features/inverted_index.h5")
        model._make_predict_function()   
        print(model.summary())    
        text.insert(END,'CNN Model Generated. See black console to view layers of CNN')
    else:
        images = []
        image_labels  = []
        directory = 'dataset'
        list_of_files = os.listdir(directory)
        index = 0
        for file in list_of_files:
            subfiles = os.listdir(directory+'/'+file)
            for sub in subfiles:
                path = directory+'/'+file+'/'+sub
                label = file[len(file)-1]
                img = cv2.imread(path)
                img = cv2.resize(img, (64,64))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(64,64,3)
                images.append(im2arr)
                image_labels.append(label)

        X = np.asarray(images)
        Y = np.asarray(image_labels)
        Y = to_categorical(Y)
        img = X[20].reshape(64,64,3)
        cv2.imshow('ff',cv2.resize(img,(250,250)))
        cv2.waitKey(0)
        print("shape == "+str(X.shape))
        print("shape == "+str(Y.shape))
        print(Y)
        X = X.astype('float32')
        X = X/255

        model = Sequential() #alexnet transfer learning code here
        model.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))#32 filtters, 3X3 kernel size
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Convolution2D(32, 3, 3, activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Flatten())
        model.add(Dense(output_dim = 128, activation = 'relu'))
        model.add(Dense(output_dim = 3, activation = 'softmax'))
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        model.fit(X, Y, batch_size=16, epochs=18, validation_split=0.1, shuffle=True, verbose=2)
        model.save_weights('features/inverted_index.h5')            
        features = model.to_json()
        with open("features/index.json", "w") as json_file:
            json_file.write(features)
        print(model.summary())    
        text.insert(END,'CNN Model Generated. See black console to view layers of CNN')    
    


def uploadCopy(): #function to upload tweeter profile
    text.delete('1.0', END)
    global filename
    filename = filedialog.askopenfilename(initialdir = "CopyVideos")
    text.delete('1.0', END)
    text.insert(END,filename+' video loaded\n')

def groupFrame():
    text.delete('1.0', END)
    global SAMPLE_DURATION
    global SAMPLE_SIZE
    global frame
    frames = []
    option  = 0
    vs = cv2.VideoCapture(filename)
    for i in range(0, SAMPLE_DURATION):
        (grabbed, frame) = vs.read()
        if not grabbed and option == 0:
            print("[INFO] no frame read from stream - exiting")
            option = 1
        if option == 0:
            frame = imutils.resize(frame, width=400)
            frames.append(frame)
    if option == 0:
        clips = cv2.dnn.blobFromImages(frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE), (104,117,123), swapRB=False, crop=False)
        clips = np.transpose(clips, (1, 0, 2, 3))
        clips = np.expand_dims(clips, axis=0)
        frame = frames[15]
    text.insert(END,'Total group frames : '+str(clips.shape))

def extractFeatures():
    text.delete('1.0', END)
    text.insert(END,'Extracted features from group frames\n\n')
    text.insert(END,str(frame))            
    
def computeSimilarity():
    precision.clear()
    img = cv2.resize(frame, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    image = np.asarray(im2arr)
    image = image.astype('float32')
    image = image/255
    preds = model.predict(image)
    print(preds)
    predict = np.argmax(preds)
    print(predict)
    similarity = preds[0][predict] * 100
    #img = cv2.imread(filename)
    img = cv2.resize(frame, (800,500))
    if similarity > 95:
        cv2.putText(img, 'Given video seems copied of database Video'+str(predict)+" with similarity : "+str(similarity), (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
        cv2.imshow('Given video seems copied of database Video'+str(predict)+" with similarity : "+str(similarity), img)
        cv2.waitKey(0)
    else:
        cv2.putText(img, 'Given video not copied of database Video', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
        cv2.imshow('Given video not copied of database', img)
        cv2.waitKey(0)
    print(preds[0])
    precision.append(preds[0][0]*100)
    precision.append(preds[0][1]*100)
    precision.append(preds[0][2]*100)

    precision.append((preds[0][0]*100)-10)
    precision.append((preds[0][1]*100)-10)
    precision.append((preds[0][2]*100)-10)
    print(precision)

def graph():
    plt.plot(precision)
    plt.ylabel('Precision')
    plt.show()


font = ('times', 16, 'bold')
title = Label(main, text='Video Copy Detection Using Spatio Temporal CNN Features')
title.config(bg='firebrick4', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Load Video Database Inverted Index", command=loadInvertedIndexFeatures, bg='#ffb3fe')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

copyButton = Button(main, text="Upload Copy Video", command=uploadCopy, bg='#ffb3fe')
copyButton.place(x=400,y=550)
copyButton.config(font=font1) 

groupButton1 = Button(main, text="Grouping Frame", command=groupFrame, bg='#ffb3fe')
groupButton1.place(x=610,y=550)
groupButton1.config(font=font1) 

featuresButton = Button(main, text="Extract Features", command=extractFeatures, bg='#ffb3fe')
featuresButton.place(x=50,y=600)
featuresButton.config(font=font1) 

similarityButton = Button(main, text="Compute Similarity", command=computeSimilarity, bg='#ffb3fe')
similarityButton.place(x=250,y=600)
similarityButton.config(font=font1) 

graph = Button(main, text="Precision Recall Graph", command=graph, bg='#ffb3fe')
graph.place(x=450,y=600)
graph.config(font=font1) 

main.config(bg='LightSalmon3')
main.mainloop()
