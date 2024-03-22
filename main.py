import tkinter as tk
import shutil
import os

import tkfilebrowser
from PIL import Image, ImageTk

from MLModel import MLModel

WIN_WIDTH = 1020
WIN_HEIGHT = 600

#TODO fixa så att man inte kan predicta innan man gjort en model 

class MainApplication(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.prediction = None
        self.image = None
        
        self.button_width = 25 #CAPS ?
        self.button_height = 3
        self.ml_model = MLModel()
        
        self.training_directories = []
        
        for d in os.listdir("Training_directories"):
            self.training_directories.append(f"Training_directories\{d}")
            
        
        # Skapa och placera knappar
        self.get_training_directories_button = tk.Button(self, text="Upload training files",height=self.button_height, width=self.button_width, command=lambda: self.get_training_directories())
        self.get_training_directories_button.grid(row=0, column=0, padx=20, pady=( 100 , 25))
        
        self.train_model_button = tk.Button(self, text="Train model", height=self.button_height, width=self.button_width, command=self.train_model)
        self.train_model_button.grid(row=1, column=0, pady=25)  
        
        self.predict_button = tk.Button(self, text="Predict", height=self.button_height, width=self.button_width, command=self.predict)
        self.predict_button.grid(row=2, column=0, pady=25) 
        
        self.upload_predict_file_button = tk.Button(self, text="Select image to predict", height=self.button_height, width=self.button_width, command=lambda: self.handle_predict_file())
        self.upload_predict_file_button.grid(row=3, column=0, pady=25)
        
        self.display_directory_buttons()

    def get_training_directories(self):
        self.training_directories_to_copy = list(tkfilebrowser.askopendirnames())
        
        for d in self.training_directories_to_copy:
            path = f'Training_directories\{os.path.basename(d)}'
            shutil.copytree(d, path)
            self.training_directories.append(path)
        
        self.display_directory_buttons()
        
    def display_directory_buttons(self):
        for i in range(len(self.training_directories)):
            button = tk.Button(self, text=os.path.basename(self.training_directories[i]))
            button.config(command=lambda b=button: self.remove_training_directory_and_button(b))
            button.grid(row=0, column=i+1, pady=25, padx=5)
            
    def remove_training_directory_and_button(self, button):
        
        #TODO Kan inte ta bort knappar som skapades i tidigare session.
        
        for path in self.training_directories:
            print(path)
            if os.path.basename(path) == button.cget("text"):
                directory_to_remove = path
                break
        
        self.training_directories.pop(self.training_directories.index(directory_to_remove))
        button.destroy() 
        shutil.rmtree(directory_to_remove)     

    def train_model(self):
        
        self.ml_model.train_and_save()
        print(self.ml_model.get_training_history())

    def predict(self):
        self.prediction = self.ml_model.get_prediction(self.image)
        self.prediction["prediction"] = os.path.basename(self.prediction["prediction"])
        self.display_prediction()
    
    def handle_predict_file(self):
        selected_file = self.select_file_to_predict()
        x_pos = 385
        y_pos = 175
        if selected_file: 
            self.image = Image.open(selected_file)
            self.image = self.image.resize((250, 250))  
            photo = ImageTk.PhotoImage(self.image)

            label = tk.Label(self,image=photo)
            label.image = photo 
            label.place(x=x_pos, y=y_pos)   
    
    def select_file_to_predict(self):
        return tkfilebrowser.askopenfilename(filetypes=[("Pictures", "*.png|*.jpg|*.JPG")], okbuttontext='Select')
    
    def display_prediction(self):
        prediction_text = tk.Label(self, text=f"Prediction: {self.prediction['prediction']} \n Accuracy: {self.prediction['confidence']}%", font=("Arial", 25))
        prediction_text.place(relx=0.5, rely=0.5, y=200, anchor="center")
        
 
if __name__ == "__main__":
    root = tk.Tk()
    root.minsize(WIN_WIDTH, WIN_HEIGHT)
    root.maxsize(WIN_WIDTH, WIN_HEIGHT)
    app = MainApplication(root)
    app.pack(side="top", fill="both", expand=True)
    root.mainloop()
