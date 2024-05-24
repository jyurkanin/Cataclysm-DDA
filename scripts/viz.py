from Xlib.display import Display
from Xlib.display import drawable
from Xlib import X

from PIL import Image
from PIL.ImageQt import ImageQt

from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers import StableDiffusionXLPipeline
import torch

from PyQt5.QtWidgets import QMainWindow, QApplication, QLineEdit, QWidget, QVBoxLayout, QComboBox, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5 import QtGui

import re
import os

torch.cuda.set_per_process_memory_fraction(0.99, 0)
torch.cuda.empty_cache()
total_memory = torch.cuda.get_device_properties(0).total_memory
print("total memory", total_memory)

# 1. Create a window
# 2. Get a handle to the CDDA window
# 3. Screenshot it when either a text box is submitted, or when a socket message arrives
# 4. Use stable diffusion to do img2img
# 5. Display the result


def getWindowByName(display, name_substring):
    screen = display.screen()
    root_window = screen.root
    window_list_atom = display.intern_atom("_NET_CLIENT_LIST")
    
    id_list = root_window.get_full_property(window_list_atom, X.AnyPropertyType)._data["value"][1]

    for wid in id_list:
        window = drawable.Window(display.display, wid)
        name = window.get_wm_name()

        if(len(name) == 0):
            if(name_substring in str(wid)):
                return window
            else:
                continue

        print(name_substring, name)
        if(name_substring in name):
            return window
    

def getScreenShot(display, target_window):
    width = target_window.get_geometry()._data["width"]
    height = target_window.get_geometry()._data["height"]

    raw = target_window.get_image(0, 0, width, height, X.ZPixmap, 0xFFFFFF)
    img = Image.frombytes("RGB", (width, height), raw.data, "raw", "RGBX")
    return img


class Visualizer(QMainWindow):
    def __init__(self):
        super().__init__()

        # stable diffusion stuff
        #self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("sdxl_img2img", torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLPipeline.from_pretrained("sdxl_text2img", torch_dtype=torch.float16)
        self.pipe.to("cuda")

        self.default_prompt = "Post apocalyptic painting by Michael Whelan."

        # X11 stuff
        self.display = Display()

        # pyqt stuff
        self.width = 1024
        self.height = 1200
        self.draw_width = 1024
        self.draw_height = 1024

        self.window = QWidget()
        
        self.window_name_combobox = QComboBox()
        self.window_name_button = QPushButton("Refresh Window List")        
        self.window_name_button.clicked.connect(self.setAllWindowNames)
        
        self.window_name_box_and_button = QHBoxLayout()
        self.window_name_box_and_button.addWidget(self.window_name_button)
        self.window_name_box_and_button.addWidget(self.window_name_combobox)

        self.createParamBoxes()

        self.prompt_textbox = QLineEdit()
        self.prompt_textbox.resize(280, 40)
        self.prompt_textbox.returnPressed.connect(self.on_prompt_submit)
        self.prompt_textbox.setText(self.default_prompt)
        
        self.image_label = QLabel()

        self.image_prompt_layout = QVBoxLayout()
        self.image_prompt_layout.addWidget(self.image_label)
        self.image_prompt_layout.addWidget(self.prompt_textbox)
        
        self.input_layout = QVBoxLayout()
        self.input_layout.addLayout(self.window_name_box_and_button)
        self.input_layout.addLayout(self.param_boxes)

        self.layout = QHBoxLayout()
        self.layout.addLayout(self.input_layout)
        self.layout.addLayout(self.image_prompt_layout)
        
        self.window.setLayout(self.layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.checkFileForUpdates)
        self.timer.start(100)

        self.prompt = ""
        
        self.window.show()

    def checkFileForUpdates(self):
        if(os.path.isfile("/tmp/saved_photo.txt")):
            self.visualizeWindow(False)
                              
    def createParamBoxes(self):
        self.param_boxes = QVBoxLayout()
        self.strength_hbox = QHBoxLayout()
        self.guidance_hbox = QHBoxLayout()
        self.inference_steps_hbox = QHBoxLayout()

        self.strength_label = QLabel("Strength")
        self.guidance_label = QLabel("Guidance")
        self.inference_steps_label = QLabel("Inference Steps")
        
        self.strength_box = QLineEdit()
        self.strength_box.setValidator(
            QtGui.QDoubleValidator(
                0.0,
                10.0,
                4,
                notation=QtGui.QDoubleValidator.StandardNotation
            )
        )
        self.strength_box.setText("0.9")
        
        self.guidance_box = QLineEdit()
        self.guidance_box.setValidator(
            QtGui.QDoubleValidator(
                0.0,
                10.0,
                4,
                notation=QtGui.QDoubleValidator.StandardNotation
            )
        )
        self.guidance_box.setText("8.0")

        self.inference_steps_box = QLineEdit()
        self.inference_steps_box.setValidator(QtGui.QIntValidator(1,100))
        self.inference_steps_box.setText("40")
        
        self.strength_hbox.addWidget(self.strength_label)
        self.strength_hbox.addWidget(self.strength_box)
        self.guidance_hbox.addWidget(self.guidance_label)
        self.guidance_hbox.addWidget(self.guidance_box)
        self.inference_steps_hbox.addWidget(self.inference_steps_label)
        self.inference_steps_hbox.addWidget(self.inference_steps_box)
        
        self.param_boxes.addLayout(self.strength_hbox)
        self.param_boxes.addLayout(self.guidance_hbox)
        self.param_boxes.addLayout(self.inference_steps_hbox)
        
    def on_prompt_submit(self):
        self.visualizeWindow(True)

    def getPrompt(self):
        f = open("saved_photo.txt")
        txt = f.read()
        f.close()
        
        os.remove("saved_photo.txt")
        
        cleaned_txt = re.sub('<.*?>', '', txt).replace("<","").replace(">","").replace("++","").replace("+","")
        cleaned_txt = cleaned_txt.replace("This photo was taken", "This painting is")
        return cleaned_txt
        
    def visualizeWindow(self, repeat):
        window = getWindowByName(self.display, self.window_name_combobox.currentText())
        if(window == None):
            return
        
        screenshot = getScreenShot(self.display, window)

        pre_prompt = self.prompt_textbox.text()
        
        if not(repeat):
            self.prompt = self.getPrompt()

        combined_prompt = pre_prompt + " " + self.prompt
            
        print("Using This Prompt:", combined_prompt)
            
        pil_img = self.generateImage(screenshot, combined_prompt)
        pixmap = QtGui.QPixmap.fromImage(ImageQt(pil_img))
        self.image_label.setPixmap(pixmap)
        
    def generateImage(self, image, prompt):
        width = image.width
        height = image.height
        
        width = 1024
        height = 1024
        
        image = image.resize((width, height))

        print("Size:", width, height)
            
        # strength is the amount of noise added to the input image
        # guidance scale is how close it matches the prompt
        return self.pipe(prompt,
                         #image=image,
                         num_inference_steps=int(self.inference_steps_box.text()),
                         #strength=float(self.strength_box.text()),
                         guidance_scale=float(self.guidance_box.text()),
                         height=height, width=width,
                         num_images_per_prompt=1).images[0]

    def setAllWindowNames(self):
        screen = self.display.screen()
        root_window = screen.root
        window_list_atom = self.display.intern_atom("_NET_CLIENT_LIST")
        
        id_list = root_window.get_full_property(window_list_atom, X.AnyPropertyType)._data["value"][1]
        
        window_names = []
        for wid in id_list:
            window = drawable.Window(self.display.display, wid)
            name = window.get_wm_name()
            if len(name) == 0:
                window_names.append(str(wid))
            else:
                window_names.append(name)

        self.window_name_combobox.addItems(window_names)


            
if __name__ == "__main__":
    app = QApplication([])
    viz = Visualizer()
    app.exec_()
