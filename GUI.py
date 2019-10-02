import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from utils import signal_processor
from model import Model

processor = signal_processor()
m = Model(name = 'sample_model',model_path = './sample_model/model.h5')

class GUI:
    """
    Class that defines graphical user interface
    """

    def __init__(self):
        """
        Window intialization and design
        """
        self.window = tk.Tk()
        self.window.title("DEMO")
        self.window.configure(background="black")
        self.window.wm_iconbitmap(r'./media/logo.ico')

        """
        Create buttons
        """
        tk.Button(self.window, text = "RECOGNIZE", width = 10, command = self.recognize).grid(row=3,column=0,sticky=tk.W)
        tk.Button(self.window, text = "RESET", width = 10, command = self.reset).grid(row = 3, column = 1, sticky = tk.W)
        tk.Button(self.window, text = "SHOW IMAGE", width = 20, command = self.show_image).grid(row = 4, column = 0, sticky = tk.W)

        """
        Create canvas
        """
        self.canvas = tk.Canvas(self.window, width = 300, height = 300, background = "white")
        self.canvas.grid(row=1,column=0, columnspan=3)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.release)

        """
        Output boxes
        """
        #create output textbox
        self.result_box = tk.Text(self.window, height=1, width = 20)
        self.result_box.grid(row=2, column=0)

        #Top 3 probs textbox
        self.top3_box = tk.Text(self.window, height=1, width = 38)
        self.top3_box.grid(row = 5, columnspan=4)

        #For collecting the input
        self.signal = []


        self.window.mainloop()

    def show_image(self):

        plt.imshow(self.img, cmap='gray')
        plt.show()

    def draw(self,event, r=3):
        self.signal.append([event.x, event.y, 0])

        x1, y1 = event.x-r, event.y-r
        x2, y2 = event.x+r, event.y+r

        self.canvas.create_oval(x1,y1,x2,y2, fill = "black")
    def release(self,event):
        self.signal.append([event.x,event.y,1])
    def reset(self):
        """
        Reset the canvas and all the results
        """
        self.signal = []
        self.result_box.delete(1.0, tk.END)
        self.top3_box.delete(1.0, tk.END)
        self.canvas.delete('all')

    def get_top3(self, predictions):
        return (predictions).argsort()[-3:]
    def recognize(self):
        '''
        Recognize what is written and display the result
        '''

        #Clear prev results
        self.result_box.delete(1.0, tk.END)
        self.top3_box.delete(1.0,   tk.END)

        #Get input image
        self.img = processor.get_image(np.array(self.signal))

        #Main prediction
        predictions = m.predict(np.expand_dims(np.expand_dims(self.img,0),-1))
        predictions = np.squeeze(predictions)
        self.result_box.insert(tk.END, "Result {}".format(np.argmax(predictions)))

        #top3 predictions
        top3 = self.get_top3(predictions)
        self.top3_box.insert(tk.END,   "Top 3: 1. {} {}%, 2. {} {}%, 3. {} {}%".format(top3[2], int(predictions[top3[2]]*100), top3[1], int(predictions[top3[1]]*100), top3[0], int(predictions[top3[0]]*100)))


GUI()
