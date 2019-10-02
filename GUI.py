import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np




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
        #tk.Button(self.window, text = "SHOW IMAGE", width = 20, command = self.show_image).grid(row = 4, column = 0, sticky = tk.W)

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



        self.signal = []


        self.window.mainloop()

    def show_image(self):
        pass
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

    def recognize(self):
        pass
