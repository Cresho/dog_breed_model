from keras.preprocessing import image
from keras.models import Model
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import customtkinter
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from PIL import Image, ImageTk

image_references = []
file_paths = []

def predict_from_image():
    img = image.load_img(file_paths[-1], target_size=(299, 299))
    img_tensor = image.img_to_array(img)                   
    img_tensor = np.expand_dims(img_tensor, axis=0)       
    img_tensor /= 255.                                     
    
    pred = model.predict(img_tensor)
    sorted_breeds_list = sorted(selected_breed_list)
    predicted_class = sorted_breeds_list[np.argmax(pred)]
    
    print(predicted_class)
    
    results = Label(frame, text=predicted_class.upper(), font=('Calibri Bold', 24), anchor=CENTER, width=23, bg="white").grid(row=10, column=0, columnspan=4)

def LoadImage():
    filetypes = (
        ('jpg files', '*.jpg'),
        ('jpeg files', '*.jpeg'),
        ('png files', '*.png')
    )

    filename = fd.askopenfilename(
        title='Select an image',
        initialdir='/',
        filetypes=filetypes
    )

    file_path = filename
    file_paths.append(file_path)
    img = Image.open(file_path)
    img = img.resize((395,395))
    photo_image = ImageTk.PhotoImage(img)
    lbImg = Label(frame, image=photo_image, anchor=CENTER).grid(row=2, column=0, rowspan=3, columnspan=3)
    image_references.append(photo_image)


num_classes = 12

model = load_model('2023-09-11_dog_breed_model.h5')
df = pd.read_csv('labels.csv')
selected_breed_list = list(df.groupby('breed').count().sort_values(by='id', ascending=False).head(num_classes).index)


root = Tk()
root.geometry("400x560")

root.title("Dog Breed Classificator")
root.resizable(False, False)

frame = Frame(root).grid()

placeholder = image.load_img("placeholder.jpg", target_size=(395, 395))
placeholderPhoto = ImageTk.PhotoImage(placeholder)

loadButton = Button(frame, text="Choose Image", font=('Calibri Bold', 10), width=26, height=2, command=LoadImage).grid(row=1, column=0, columnspan=2, padx=5, sticky=W)
classificateButton = Button(frame, text="Generate Results", font=('Calibri Bold', 10), width=26, height=2, command=predict_from_image).grid(row=1, column=2, columnspan=2, padx=0, sticky=W)
blankLabel = Label(frame, image=placeholderPhoto, anchor=CENTER).grid(row=2, column=0, rowspan=3,columnspan=3)
resultLabel = Label(frame, text="Results:", font=('Calibri Bold', 24), anchor=CENTER).grid(row=9, column=0, columnspan=4)

root.mainloop()