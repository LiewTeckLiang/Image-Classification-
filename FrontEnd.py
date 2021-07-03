from tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2
from tkinter import filedialog
from Testing import *


def classify(path):
    # READ IMAGE
    imgOriginal = cv2.imread(path)

    # PROCESS IMAGE
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (64, 64))
    img = preprocessing(img)
    img = img.reshape(1, 64, 64, 1)

    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue = np.amax(predictions)
    if probabilityValue > threshold:
        return str(getClassName(classIndex)), probabilityValue


def select_image():
    # grab a reference to the image panels
    global panelA, label

    # open a file chooser dialog and allow the user to select an input
    # image

    path = filedialog.askopenfilename()


    # ensure a file path was selected

    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image = cv2.imread(path)

        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 320))
       
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        classify(path)
        answer, probability = classify(path)

        # Initialize the panel
        if panelA is None or label is None:
            # the first panel will store our original image
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="bottom", padx=10, pady=10)

            label = Label(root, text=answer)
            label.pack(side="top", fill="both", expand="yes", padx="10", pady="10")

         
        # otherwise, update the image panels
        else:
            # update the panels
            panelA.configure(image=image)
            panelA.image = image

            label.config(text=answer)

           

# initialize the window toolkit along with the two image panels
root = Tk()
panelA = None
label = None
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
root.mainloop()
