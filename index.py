import numpy as np
import cv2 as cv
import tkinter as tk 
from tkinter import messagebox
import os
import pymongo

window = tk.Tk()
window.title("FACE RECOGNITION SYSTEM")

l1 = tk.Label(window, text="Id", font=('American Typewriter', 25))
l1.grid(column=0, row=0)  
t1 = tk.Entry(window, width=50, bd=2)
t1.grid(column=1, row=0)

l2 = tk.Label(window, text="Name", font=('American Typewriter', 25))
l2.grid(column=0, row=1)
t2 = tk.Entry(window, width=50, bd=2)
t2.grid(column=1, row=1)

l3 = tk.Label(window, text="Address", font=('American Typewriter', 25))
l3.grid(column=0, row=2)
t3 = tk.Entry(window, width=50, bd=2)
t3.grid(column=1, row=2)


def generate_data():
    if not os.path.exists("faces"):
        os.mkdir("faces")
    
    if (t1.get() == "" or t2.get() == "" or t3.get() == ""):
        messagebox.showinfo("Result", "Please provide all the information of the User!!")
        return
    
    mongo_uri = "mongodb://localhost:27017/face_recognizer"
    client = pymongo.MongoClient(mongo_uri)
    db = client["my_database"]
    collection = db["id"]
    
    user_id = int(t1.get())
    existing_user = collection.find_one({"id": user_id})
    
    if existing_user:
        result = messagebox.askquestion("ID Exists", f"ID '{user_id}' already exists. Do you want to replace the data?")
        if result == 'yes':
            collection.update_one({"id": user_id}, {"$set": {"name": t2.get(), "address": t3.get()}})
            
            # Delete old images
            for file in os.listdir("faces"):
                if file.startswith(f"{user_id}_"):
                    os.remove(os.path.join("faces", file))
        else:
            messagebox.showinfo("Info", "Please choose a different ID.")
            client.close()
            return
    else:
        collection.insert_one({
            "id": user_id,
            "name": t2.get(),
            "address": t3.get()
        })
    
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    while count < 400:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
        for (x,y,w,h) in faces:
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            count += 1
            face_image = gray[y:y+h, x:x+w]
            cv.imwrite(f'faces/{user_id}_{count}.jpg', face_image)
    
        cv.putText(frame, str(count), (frame.shape[1] // 2, frame.shape[0] - 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv.imshow('Captured Image', frame)
            
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    messagebox.showinfo("Result", "Data Collection Completed!")
    
    cap.release()
    cv.destroyAllWindows()
    
    train_classifier()

def train_classifier():
    faces = os.getcwd() + "/faces/"
    path = [os.path.join(faces, f) for f in os.listdir(faces) if f.endswith(('.jpg', '.png', '.jpeg'))]

    face = []
    ids = []

    for image in path:
        img = cv.imread(image, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load image: {image}")
            continue
        
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[-1].split('_')[0])

        face.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    clf = cv.face.LBPHFaceRecognizer_create()
    clf.train(face, ids)
    clf.write("trained_classifier.xml")

    messagebox.showinfo("Result", "Training data completed!")

def detect_face():
    def draw_boundary(img, classifier, scalefactor, minNeighbors, color, clf):
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        face = classifier.detectMultiScale(gray_img, scalefactor, minNeighbors)
    
        for (x, y, w, h) in face:
            cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
            id, confidence = clf.predict(gray_img[y:y+h, x:x+w])
    
            mongo_uri = "mongodb://localhost:27017/face_recognizer"
            client = pymongo.MongoClient(mongo_uri)
            db = client["my_database"]
            collection = db["id"]

            user_data = collection.find_one({"id": id})
            name = user_data.get("name", "UNKNOWN") if user_data else "UNKNOWN"
    
            if confidence < 50:  
                cv.putText(img, name, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            else:
                cv.putText(img, "UNKNOWN", (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    
        return img  

    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    clf = cv.face.LBPHFaceRecognizer_create()
    clf.read("trained_classifier.xml")
    
    cap = cv.VideoCapture(0) 
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    
    while True:
        ret, img = cap.read()
        img = draw_boundary(img, face_cascade, 1.3, 5, (255, 255, 255), clf)
        cv.imshow('Detected face', img)
    
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

b1 = tk.Button(window, text="Generate Data", font=('American Typewriter', 40), bg="green", fg="orange", command=generate_data)
b1.grid(column=0, row=4)

b2 = tk.Button(window, text="Training", font=('American Typewriter', 40), bg="green", fg="orange", command=train_classifier)
b2.grid(column=1, row=4)

b3 = tk.Button(window, text="Detect Face", font=('American Typewriter', 40), bg="green", fg="orange", command=detect_face)
b3.grid(column=2, row=4)

window.geometry("1100x200")
window.mainloop()
