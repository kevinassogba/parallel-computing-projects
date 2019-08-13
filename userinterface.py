from tkinter import *
from tkinter import messagebox
from tkinter import font
from PIL import Image, ImageTk

import classify

# The very definition of this era in "technological outburst".

class tellmeapp:

    __Slots__ = "output_label"

    def initialize(self):

        screen = Tk()
        screen.title("Tell Me Who")

        canvas = Canvas(screen, height = 500, width = 600)
        canvas.pack()

        image = Image.open("./library.jpg")
        bg_image = ImageTk.PhotoImage(image)
        bg_im_label = Label(screen, image = bg_image)
        bg_im_label.place(relwidth = 1, relheight = 1)

        """
            Instruction frame
        """

        inst_frame = Frame(screen, bg = "#c9c930", bd = 5)
        inst_frame.place(relx = 0.1, rely = 0.1, relheight = 0.5, relwidth = 0.8 )

        input_entry = Text(inst_frame, bd = 2, font = ('Modern', 10), insertbackground = "white" )
        input_entry.place(relheight = 0.83, relwidth = 0.75)

        radio_frame = Frame(inst_frame, bd = 2)
        radio_frame.place(relx = 0.76, relheight = 0.83, relwidth = 0.24)
        var = IntVar()
        R1 = Radiobutton(inst_frame, text = "Decision tree", font = ('Modern', 9), variable = var, value = 1, selectcolor = "black")
        R1.place(relx = 0.76, rely = 0.05)
        R2 = Radiobutton(inst_frame, text = "Logistic", font = ('Modern', 9), variable = var, value = 2, selectcolor = "black")
        R2.place(relx = 0.76, rely = 0.25)
        R3 = Radiobutton(inst_frame, text = "Both models", font = ('Modern', 9), variable = var, value = 3, selectcolor = "black")
        R3.place(relx = 0.76, rely = 0.45)
        R4 = Radiobutton(inst_frame, text = "Best model", font = ('Modern', 9), variable = var, value = 4, selectcolor = "black")
        R4.place(relx = 0.76, rely = 0.65)

        run_btn = Button(inst_frame, text = "Run", bg = "#80c1ff", command = lambda: self.tellmewho(input_entry.get('1.0','end-1c'), var.get()))
        run_btn.place(relx = 0.45, rely = 0.85)


        # testAusten1.txt
        """
            Result frame
        """

        res_frame = Frame(screen, bg = "#f4f4e5", bd = 5)
        res_frame.place(relx = 0.1, rely = 0.62, relheight = 0.35, relwidth = 0.8)

        self.output_label = Label(res_frame)
        self.output_label.place(relwidth = 1, relheight = 1)

        screen.mainloop()

    # Functions

    def tellmewho(self, file, method):
        output = classify.main(file, method)
        self.output_label.config(text = output)


def main():
    app = tellmeapp()
    app.initialize()

main()
