import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, simpledialog
from PIL import Image
import numpy as np
import json
import os
import cv2
import re
import uuid
import cvzone
import threading
import pandas as pd
import requests
from ultralytics import YOLO

def geometry_center(app_width, app_height):    
    screen_width = app.winfo_screenwidth()
    screen_height = app.winfo_screenheight()
    
    x = int((screen_width - app_width)/2)
    y = int((screen_height - app_height)/2)
    
    center = f"{app_width}x{app_height}+{x}+{y}"
    
    return center

def manual_book():
    modal = ctk.CTkToplevel(app)
    modal.title("")
    modal.geometry(geometry_center(300, 250))
    modal.resizable(False, False)
    
    frame_header = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_header.pack(fill="x")
    
    frame_body = ctk.CTkFrame(modal, bg_color="#27272a", fg_color="#27272a")
    frame_body.pack(expand="true", fill="both")
    
    frame_btn = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_btn.pack(fill="x")

    # Memuat dan mengubah ukuran gambar menggunakan PIL
    image_path = "images/icons/form.png"
    image = Image.open(image_path)
    resized_image = image.resize((20, 20))
    
    image_ctk = ctk.CTkImage(light_image=resized_image, size=(20, 20))

    image_header = ctk.CTkLabel(frame_header, image=image_ctk, text="")
    image_header.pack(side="left", fill="x", padx=(20,0), pady=(20))
    
    header = ctk.CTkLabel(frame_header, text="Manual Book", font=("Helvetica", 15, "bold"))
    header.pack(side="left", fill="x", padx=10, pady=20) 
    
    style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"))
    
    tree = ttk.Treeview(frame_body, columns=("#", "Key", "Function"), show="headings", height=0)
    tree.heading("#", text="#")
    tree.heading("Key", text="Key")
    tree.heading("Function", text="Function")
    
    tree.column("#", width=30, stretch=False, anchor="center")
    tree.column("Key", width=80, stretch=True)
    tree.column("Function", stretch=True)
    
    style.configure("Treeview", rowheight=25, font=("Helvetica", 10), foreground="#1f2937")
    style.map("Treeview", background=[("selected", "#4ade80")], foreground=[("selected", "#000")])
    tree.tag_configure("evenrow", background="#d1d5db")
    tree.tag_configure("oddrow", background="#e5e7eb")

    
    datas = [
        {
            "key":"i",
            "function":"Show manual book"
        },
        {
            "key":"left click",
            "function":"Draw the polyline"
        },
        {
            "key":"right click",
            "function":"Delete the polyline"
        },
        {
            "key":"s",
            "function":"Save"
        },
        {
            "key":"q",
            "function":"Quit"
        },
    ]
    
    for i, data in enumerate(datas):
        index = i+1
        values = (index, data["key"], data["function"])
        if i % 2 == 0:
            tree.insert("", "end", values=values, tags="evenrow")
        else:
            tree.insert("", "end", values=values, tags="oddrow")
        
    tree.pack(expand="true", fill="both", padx=10, pady=10)
    
    def close():
        modal.destroy()
        
    modal.protocol("WM_DELETE_WINDOW", lambda: close())
    
    btn = ctk.CTkButton(frame_btn, text="Hide", width=60, fg_color="#4ade80", text_color="#000", hover_color="#16a34a",command=lambda: close())
    btn.pack(side="right", padx=(10), pady=10)
    
    modal.transient(app)
    modal.grab_set()
    modal.focus()
    app.wait_window(modal)

app = ctk.CTk()
app.title("Parking Area Detection")

ctk.set_appearance_mode("dark")

style = ttk.Style()

# frame
# frame header
frame_header = ctk.CTkFrame(app, bg_color="#09090b", fg_color="#09090b")
frame_header.pack(fill="x")
frame_header.grid_columnconfigure(0, weight=1)

frame_body = ctk.CTkFrame(app, bg_color="#27272a", fg_color="#27272a")
frame_body.pack(expand="true", fill="both")

frame_action = ctk.CTkFrame(frame_body, bg_color="#27272a", fg_color="#27272a")
frame_action.pack(fill="x")

frame_content = ctk.CTkFrame(frame_body, bg_color="#27272a", fg_color="#27272a")
frame_content.pack(expand="true", fill="both", padx=10, pady=10)
frame_content.grid_columnconfigure(1, weight=1)

manual_book()

app.geometry(geometry_center(480,300))
app.mainloop()