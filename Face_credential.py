from tkinter import *
import tkinter as tk
import sqlite3
import numpy as np
import cv2
from PIL import Image
import os

def register_face():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("drop table users")
    c.execute('''CREATE TABLE users (id integer unique primary key autoincrement,name text)''')
    print('Table created !')
    conn.commit()
    conn.close()

    global var_for_saving_name
    # making a folder for storing custom based image data
    conn = sqlite3.connect('database.db')
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')
    c = conn.cursor()

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # to capture frames from the webcam of laptop
    cam = cv2.VideoCapture(0)
    username = new_entry.get()
    print(username)
    c.execute("INSERT INTO users (name) VALUES (?)", (username,))
    uid = c.lastrowid
    sample = 0
    while True:
        check, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converting each frame to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sample = sample + 1
            cv2.imwrite("dataset/User." + str(uid) + "." + str(sample) + ".jpg", gray[y:y + h, x:x + w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.waitKey(100)
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)
        if sample > 25:  # will store 25 frames
            break
    cam.release()
    conn.commit()
    conn.close()
    cv2.destroyAllWindows()

    from PIL import Image
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path = 'dataset'
    if not os.path.exists('./recognizer'):
        os.makedirs('./recognizer')

    def getImagesWithID(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faces = []
        IDs = []
        for imagePath in imagePaths:
            faceImg = Image.open(imagePath).convert('L')
            faceNp = np.array(faceImg, 'uint8')
            ID = int(os.path.split(imagePath)[-1].split('.')[1])
            faces.append(faceNp)
            IDs.append(ID)
            cv2.imshow("training", faceNp)
            cv2.waitKey(10)
        return np.array(IDs), faces
    Ids, faces = getImagesWithID(path)
    recognizer.train(faces, Ids)
    recognizer.save('recognizer/trainingData.yml')
    print('Process Finished')
    cv2.destroyAllWindows()
    label = Label(root, image=finger)
    label.place(x=340, y=70)

def recog_me():

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    fname = "recognizer/trainingData.yml"
    if not os.path.isfile(fname):
        print("Please train the data first")
        exit(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(fname)
    while True:
        check, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 0), 3)
            ids, conf = recognizer.predict(gray[y:y + h, x:x + w])
            c.execute("select name from users where id = (?);", (ids,))
            result = c.fetchall()
            name = result[0][0]
            if conf < 50:
                cv2.putText(frame, name, (x + 2, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Match', (x + 2, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Face Recognizer', frame)
        k = cv2.waitKey(30)
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()


root = Tk()
root.title('Face Lock')
root.geometry('800x400')
root.configure(background='white')
root.resizable(0,0)

cop = Image.open('cop.png')
cop = cop.resize((35,35),Image.ANTIALIAS)
cop.save("img.ppm","ppm")
cop = PhotoImage(file='img.ppm')

finger = Image.open('d_finger.jpg')
finger = finger.resize((90,90),Image.ANTIALIAS)
finger= finger.save('d_finger.ppm','ppm')
finger = PhotoImage(file='d_finger.ppm')

title = Label(root, text='Welcome to Face Credential Database System', fg='black',bg='white',font=('bold',17))
title.place(x=200,y=5)
pic = Label(root,image=cop)
pic.place(x=160,y=0)

new_user = Label(root, text='Enter Name',bg='black',fg='white')
new_user.place(x=10,y=110)
new_entry = Entry(root,width=18,bg='LightBlue1')
new_entry.place(x=90,y=110)

reg_button = Button(root,text='Register your face', activebackground='gold',bg='orange',command=register_face)
reg_button.place(x=50,y=200)

recog = Button(root,text='Verify entry',activebackground='gold',bg='orange',command=recog_me)
recog.place(x=350,y=200)
credits = Label(root,bg='black',fg='white',text='©Developed by Bhavishya Pandit',height=3,width=120)
credits.place(x=0,y=350)

root.mainloop()