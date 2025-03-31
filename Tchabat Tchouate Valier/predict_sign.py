import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import json
import os

IMAGE_SIZE = (30, 30)

class TrafficSignClassifier:
    def __init__(self, model_file):
        self.model = tf.keras.models.load_model(model_file)
        categories_path = os.path.splitext(model_file)[0] + "_categories.json"
        
        with open(categories_path, "r") as f:
            self.categories = json.load(f)
        
        self.root = tk.Tk()
        self.root.title("Traffic Sign Classifier")
        self.root.geometry("1000x600")
        
        # Layout frames
        self.left_panel = ttk.Frame(self.root, padding="10")
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_panel = ttk.Frame(self.root, padding="10")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
        
        # Left panel widgets
        ttk.Label(self.left_panel, text="Traffic Sign Recognition", font=('Helvetica', 16, 'bold')).pack(pady=10)
        ttk.Button(self.left_panel, text="Select Image", command=self.load_image).pack(pady=10)
        
        self.image_display = ttk.Label(self.left_panel)
        self.image_display.pack(pady=10)
        
        # Right panel widgets
        self.result_container = ttk.LabelFrame(self.right_panel, text="Prediction", padding="10")
        self.result_container.pack(fill=tk.BOTH, expand=True)
        
        self.category_info = ttk.Label(self.result_container, text="Category: ")
        self.category_info.pack(pady=5, anchor=tk.W)
        
        self.sign_info = ttk.Label(self.result_container, text="Sign: ")
        self.sign_info.pack(pady=5, anchor=tk.W)
        
        self.confidence_info = ttk.Label(self.result_container, text="Confidence: ")
        self.confidence_info.pack(pady=5, anchor=tk.W)
    
    def classify_sign(self, img_path):
        """Processes the image and predicts the traffic sign."""
        image = cv2.imread(img_path)
        if image is None:
            return None
        
        h, w = image.shape[:2]
        scale = min(IMAGE_SIZE[0] / w, IMAGE_SIZE[1] / h)
        new_dim = (int(w * scale), int(h * scale))
        resized_img = cv2.resize(image, new_dim)
        
        # Create blank canvas and center the resized image
        processed_img = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
        x_offset = (IMAGE_SIZE[0] - new_dim[0]) // 2
        y_offset = (IMAGE_SIZE[1] - new_dim[1]) // 2
        processed_img[y_offset:y_offset + new_dim[1], x_offset:x_offset + new_dim[0]] = resized_img
        
        # Prepare input
        processed_img = np.expand_dims(processed_img, axis=0)
        
        # Model prediction
        prediction = self.model.predict(processed_img, verbose=0)[0]
        best_match = np.argmax(prediction)
        confidence = float(prediction[best_match])
        label = self.categories.get(str(best_match), "Unknown")
        
        return best_match, label, confidence
    
    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            img = Image.open(path)
            img.thumbnail((400, 400))
            img_display = ImageTk.PhotoImage(img)
            self.image_display.config(image=img_display)
            self.image_display.image = img_display
            
            result = self.classify_sign(path)
            if result:
                best_match, label, confidence = result
                self.category_info.config(text=f"Category: {best_match}")
                self.sign_info.config(text=f"Sign: {label}")
                self.confidence_info.config(text=f"Confidence: {confidence:.3%}")
    
    def start(self):
        self.root.mainloop()

if __name__ == "__main__":
    MODEL_FILE = "best_model.h5" 
    classifier = TrafficSignClassifier(MODEL_FILE)
    classifier.start()
