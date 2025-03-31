import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os

IMAGE_SIZE = (30, 30)

class TrafficSignClassifier:
    def __init__(self, model_file):
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
            
        print(f"Loading model from {model_file}")
        self.model = tf.keras.models.load_model(model_file)
        
        self.root = tk.Tk()
        self.root.title("Traffic Sign Classifier")
        self.root.geometry("1200x700")
        self.root.configure(bg='#f0f0f0')
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure('Modern.TFrame', background='#f0f0f0')
        self.style.configure('Modern.TButton',
                           font=('Helvetica', 12),
                           padding=10,
                           background='#2196F3')
        self.style.configure('Title.TLabel',
                           font=('Helvetica', 24, 'bold'),
                           background='#f0f0f0',
                           foreground='#1976D2')
        self.style.configure('Info.TLabel',
                           font=('Helvetica', 12),
                           background='#ffffff',
                           padding=5)
        
        # Main container
        self.main_container = ttk.Frame(self.root, style='Modern.TFrame')
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_frame = ttk.Frame(self.main_container, style='Modern.TFrame')
        title_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(title_frame,
                 text="Traffic Sign Recognition",
                 style='Title.TLabel').pack()
        
        # Content container with shadow effect
        self.content_frame = tk.Frame(self.main_container,
                                    bg='#ffffff',
                                    relief='solid',
                                    borderwidth=1)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel (Image display)
        self.left_panel = ttk.Frame(self.content_frame, style='Modern.TFrame')
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Image placeholder
        self.image_frame = tk.Frame(self.left_panel, bg='#f5f5f5')
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.image_display = ttk.Label(self.image_frame)
        self.image_display.pack(pady=20)
        
        self.upload_btn = ttk.Button(self.left_panel,
                                   text="Select Image",
                                   command=self.load_image,
                                   style='Modern.TButton')
        self.upload_btn.pack(pady=20)
        
        # Right panel (Results)
        self.right_panel = ttk.Frame(self.content_frame, style='Modern.TFrame')
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=20, pady=20)
        
        # Results container
        self.result_container = tk.Frame(self.right_panel,
                                       bg='#ffffff',
                                       relief='solid',
                                       borderwidth=1)
        self.result_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(self.result_container,
                 text="Recognition Results",
                 style='Title.TLabel',
                 font=('Helvetica', 18, 'bold')).pack(pady=20)
        
        # Results info with modern styling
        self.category_info = ttk.Label(self.result_container,
                                     text="Category: Waiting for image...",
                                     style='Info.TLabel')
        self.category_info.pack(pady=10, padx=20, fill=tk.X)
        
        self.sign_info = ttk.Label(self.result_container,
                                  text="Sign: Waiting for image...",
                                  style='Info.TLabel')
        self.sign_info.pack(pady=10, padx=20, fill=tk.X)
        
        self.confidence_frame = tk.Frame(self.result_container, bg='#ffffff')
        self.confidence_frame.pack(pady=10, padx=20, fill=tk.X)
        
        self.confidence_info = ttk.Label(self.confidence_frame,
                                       text="Confidence: 0%",
                                       style='Info.TLabel')
        self.confidence_info.pack(side=tk.LEFT)
        
        # Progress bar for confidence
        self.confidence_bar = ttk.Progressbar(self.confidence_frame,
                                            length=200,
                                            mode='determinate')
        self.confidence_bar.pack(side=tk.RIGHT, padx=10)
    
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
        
        return best_match, confidence
    
    def update_results(self, category, confidence):
        self.category_info.config(text=f"Category: {category}")
        self.sign_info.config(text=f"Sign Type: {self.get_sign_description(category)}")
        self.confidence_info.config(text=f"Confidence: {confidence:.1%}")
        self.confidence_bar['value'] = confidence * 100
        
    def get_sign_description(self, category):
        signs = {
            0: "Speed limit (20km/h)",
            1: "Speed limit (30km/h)",
            2: "Speed limit (50km/h)",
            3: "Speed limit (60km/h)",
            4: "Speed limit (70km/h)",
            5: "Speed limit (80km/h)",
            6: "End of speed limit (80km/h)",
            7: "Speed limit (100km/h)",
            8: "Speed limit (120km/h)",
            9: "No passing",
            10: "No passing for vehicles over 3.5 metric tons",
            11: "Right-of-way at the next intersection",
            12: "Priority road",
            13: "Yield",
            14: "Stop",
            15: "No vehicles",
            16: "Vehicles over 3.5 metric tons prohibited",
            17: "No entry",
            18: "General caution",
            19: "Dangerous curve to the left",
            20: "Dangerous curve to the right",
            21: "Double curve",
            22: "Bumpy road",
            23: "Slippery road",
            24: "Road narrows on the right",
            25: "Road work",
            26: "Traffic signals",
            27: "Pedestrians",
            28: "Children crossing",
            29: "Bicycles crossing",
            30: "Beware of ice/snow",
            31: "Wild animals crossing",
            32: "End of all speed and passing limits",
            33: "Turn right ahead",
            34: "Turn left ahead",
            35: "Ahead only",
            36: "Go straight or right",
            37: "Go straight or left",
            38: "Keep right",
            39: "Keep left",
            40: "Roundabout mandatory",
            41: "End of no passing",
            42: "End of no passing by vehicles over 3.5 metric tons"
        }
        return signs.get(category, f"Unknown Sign (Category {category})")
    
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ppm")])
        if path:
            # Show loading state
            self.upload_btn.config(state='disabled')
            self.category_info.config(text="Processing...")
            self.root.update()
            
            try:
                # Load and display image
                img = Image.open(path)
                # Calculate aspect ratio for resizing
                display_size = (350, 350)
                img.thumbnail(display_size, Image.Resampling.LANCZOS)
                img_display = ImageTk.PhotoImage(img)
                self.image_display.config(image=img_display)
                self.image_display.image = img_display
                
                # Classify image
                result = self.classify_sign(path)
                if result:
                    self.update_results(*result)
            except Exception as e:
                self.category_info.config(text=f"Error: {str(e)}")
            finally:
                self.upload_btn.config(state='normal')
    
    def start(self):
        self.root.mainloop()

if __name__ == "__main__":
    MODEL_FILE = "best_model.h5" 
    classifier = TrafficSignClassifier(MODEL_FILE)
    classifier.start()
