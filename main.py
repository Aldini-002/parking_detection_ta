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

def modal_okcancel(message):
    result = [None]
    modal = ctk.CTkToplevel(app)
    modal.title("")
    modal.geometry(geometry_center(300, 180))
    modal.resizable(False, False)
    
    frame_header = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_header.pack(fill="x")
    
    frame_msg = ctk.CTkFrame(modal, bg_color="#27272a", fg_color="#27272a")
    frame_msg.pack(expand="true", fill="both")
    
    frame_btn = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_btn.pack(fill="x")
    frame_btn.grid_columnconfigure((0,1), weight=1)

    # Memuat dan mengubah ukuran gambar menggunakan PIL
    image_path = "images/icons/warning.png"
    image = Image.open(image_path)
    resized_image = image.resize((20, 20))
    
    # Menggunakan CTkImage untuk memastikan kompatibilitas HighDPI
    image_ctk = ctk.CTkImage(light_image=resized_image, size=(20, 20))

    image_header = ctk.CTkLabel(frame_header, image=image_ctk, text="")
    image_header.pack(side="left", fill="x", padx=(20,0), pady=(20))
    
    header = ctk.CTkLabel(frame_header, text="WARNING", font=("Helvetica", 15, "bold"))
    header.pack(side="left", fill="x", padx=10, pady=20) 
    
    msg = ctk.CTkLabel(frame_msg, text=message, font=("Helvetica", 15), anchor="w")
    msg.pack(expand="true", fill="both", padx=40)
    
    def close(value):
        result[0] = value
        modal.destroy()
        
    modal.protocol("WM_DELETE_WINDOW", lambda: close(None))
    
    btn = ctk.CTkButton(frame_btn, text="Cancel", width=60, fg_color="#4ade80", text_color="#000", hover_color="#16a34a",command=lambda: close(None))
    btn.pack(side="right", padx=(10), pady=10)
    
    btn = ctk.CTkButton(frame_btn, text="Ok", width=60, fg_color="#27272a", border_width=1, border_color="#f87171", hover_color="#dc2626", command=lambda: close("yes"))
    btn.pack(side="right", padx=0, pady=10)
    
    
    modal.transient(app)
    modal.grab_set()
    modal.focus()
    app.wait_window(modal)
    
    return result[0]
    
def modal_input(message):
    result = [None]

    modal = ctk.CTkToplevel(app)
    modal.title("")
    modal.geometry(geometry_center(300, 180))
    modal.resizable(False, False)
    
    frame_header = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_header.pack(fill="x")
    
    frame_entry = ctk.CTkFrame(modal, bg_color="#27272a", fg_color="#27272a")
    frame_entry.pack(expand="true", fill="both")
    frame_entry.grid_columnconfigure(1, weight=1)
    frame_entry.grid_rowconfigure(0, weight=1)
    
    frame_btn = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_btn.pack(fill="x")
    frame_btn.grid_columnconfigure((0,1), weight=1)

    # Memuat dan mengubah ukuran gambar menggunakan PIL
    image_path = "images/icons/form.png"
    image = Image.open(image_path)
    resized_image = image.resize((20, 20))
    
    # Menggunakan CTkImage untuk memastikan kompatibilitas HighDPI
    image_ctk = ctk.CTkImage(light_image=resized_image, size=(20, 20))

    image_header = ctk.CTkLabel(frame_header, image=image_ctk, text="")
    image_header.pack(side="left", fill="x", padx=(20,0), pady=(20))
    
    header = ctk.CTkLabel(frame_header, text="FORM", font=("Helvetica", 15, "bold"))
    header.pack(side="left", fill="x", padx=10, pady=20) 
    
    input_label = ctk.CTkLabel(frame_entry, text=message, font=("Helvetica", 12))
    input_label.grid(row=0, column=0, padx=(30,5), sticky="e")
    
    input_entry = ctk.CTkEntry(frame_entry)
    input_entry.grid(row=0, column=1, padx=(0, 30), sticky="we")
    
    def close(value):
        result[0] = value
        modal.destroy()
        
    modal.protocol("WM_DELETE_WINDOW", lambda: close(None))
    
    btn = ctk.CTkButton(frame_btn, text="Ok", width=60, fg_color="#4ade80", text_color="#000", hover_color="#16a34a", command=lambda: close(input_entry.get()))
    btn.pack(side="right", padx=10, pady=10)
    
    btn = ctk.CTkButton(frame_btn, text="Cancel", width=60, fg_color="#27272a", border_width=1, border_color="#f87171", hover_color="#dc2626", command=lambda: close(None))
    btn.pack(side="right", padx=(0), pady=10)
    
    modal.transient(app)
    modal.grab_set()
    modal.focus()
    app.wait_window(modal)
    
    return result[0]
    
def modal_alert(status, message):
    modal = ctk.CTkToplevel(app)
    modal.title("")
    modal.geometry(geometry_center(300, 180))
    modal.resizable(False, False)
    
    frame_header = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_header.pack(fill="x")
    
    frame_msg = ctk.CTkFrame(modal, bg_color="#27272a", fg_color="#27272a")
    frame_msg.pack(expand="true", fill="both")
    
    frame_btn = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_btn.pack(fill="x")

    # Memuat dan mengubah ukuran gambar menggunakan PIL
    image_path = "images/icons/warning.png"
    if status == "success":
        image_path = "images/icons/success.png"
    elif status == "failed":
        image_path = "images/icons/failed.png"
    image = Image.open(image_path)
    resized_image = image.resize((20, 20))
    
    # Menggunakan CTkImage untuk memastikan kompatibilitas HighDPI
    image_ctk = ctk.CTkImage(light_image=resized_image, size=(20, 20))

    image_header = ctk.CTkLabel(frame_header, image=image_ctk, text="")
    image_header.pack(side="left", fill="x", padx=(20,0), pady=(20))
    
    header = ctk.CTkLabel(frame_header, text=status.upper(), font=("Helvetica", 15, "bold"))
    header.pack(side="left", fill="x", padx=10, pady=20) 
    
    msg = ctk.CTkLabel(frame_msg, text=message, font=("Helvetica", 13), wraplength=300)
    msg.pack(expand="true", fill="both", padx=20)
    
    def close():
        modal.destroy()
    
    btn = ctk.CTkButton(frame_btn, text="Ok", width=60, fg_color="#4ade80", text_color="#000", hover_color="#16a34a",command=lambda: close())
    btn.pack(side="right", padx=(10), pady=10)
    
    modal.transient(app)
    modal.grab_set()
    modal.focus()
    app.wait_window(modal)

def folder_check(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder not found : {folder_path}")
        try:
            os.mkdir(folder_path)
            print(f"Folder created : {folder_path}")
        except Exception as e:
            print(f"Folder failed to create")
        return False
    else:
        return True

# file check
def file_check(file_path):
    if not os.path.isfile(file_path):
        print(f"File not found : {file_path}")
        return False
    else:
        return True

def save_data(data):
    with open(data["file_path"], 'w') as f:
        json.dump(data, f)
    
# get data
def get_data(file_path):
    if file_check(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

# get all data
def get_all_data():
    datas = []
    folder_path = "data/parking_areas"
    if folder_check(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".json"):
                file_path = os.path.join(folder_path, file_name)
                data = get_data(file_path)
                datas.append(data)
    return datas

def all_area_name():
    area_name = []
    datas = get_all_data()
    
    for data in datas:
        area_name.append(data["area_name"])
    
    return area_name

def all_area_names():
    area_names = []
    datas = get_all_data()
    
    for data in datas:
        if "area_names" in data:
            area_names.extend(data["area_names"])
    
    return area_names

# select row
def select_row():
    data = {}
    select_item = tree.focus()
    if select_item:
        item = tree.item(select_item)
        values = item["values"]
        print(values)
        if values:
            i, area_name, source, total_area, file_name, file_path = values
            data = {
                "area_name":area_name,
                "source":source,
                "file_name":file_name,
                "file_path":file_path
            }
        else:
            modal_alert("warning", "Values is null")
    else:
        modal_alert("warning", "No item selected")
    return data

# select multi row
def select_multi_row():
    ids = []
    datas = []
    
    select_items = tree.selection()
    
    if select_items:
        for id in select_items:
            ids.append(id)
        for id in ids:
            item = tree.item(id)
            values = item["values"]
            if values:
                i, area_name, source, total_area, file_name, file_path = values
                datas.append(
                    {
                        "area_name":area_name,
                        "source":source,
                        "file_name":file_name,
                        "file_path":file_path
                    }
                )
            else:
                modal_alert("warning", "Values is null")
    else:
        modal_alert("warning", "No item selected")
    return datas

# header_title
def header_title(title):
    header_title = ctk.CTkLabel(frame_header, text=title, font=("Helvetica", 20, "bold"))
    header_title.pack(side="top", ipady=20, fill="x")

# clear widget
def clear_widget(frame):
    for widget in frame.winfo_children():
        widget.destroy()

def setup_parking(area_name, source):
    play_video = source
    if source.isdigit():
        play_video = int(source)
        
    video = cv2.VideoCapture(play_video)  # video parkir
    
    if not video.isOpened():
        modal_alert("failed", "Failed to start video")
        return

    app.withdraw()
    
    area_id = str(uuid.uuid4())
    file_name = f'{area_name.replace(" ", "_")}.json'
    file_path = f"data/parking_areas/{file_name}"
    area_names = []
    polylines = []
    
    coordinates = []
    drawing = False
    modal_active = False

    def draw_parking_mapping(event, x, y, flags, param):
        nonlocal polylines, area_names, coordinates, drawing, modal_active
        
        if modal_active:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            coordinates = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                coordinates.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if coordinates:
                modal_active = True
                current_name = modal_input("Area name")
                modal_active = False
                if current_name:
                    current_name = current_name.upper()
                    if current_name in all_area_names() or current_name in area_names:
                        modal_active = True
                        modal_alert("warning", f"{current_name} already exist")
                        modal_active = False
                    else:
                        area_names.append(current_name)
                        polylines.append(np.array(coordinates, np.int32))
                coordinates = []
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, polyline in enumerate(polylines):
                polyline_target = cv2.pointPolygonTest(polyline, (x, y), False)
                if polyline_target >= 0:
                    del polylines[i]
                    del area_names[i]
                    break

    while True:
        ret, frame = video.read()
        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (720, 480))

        for i, polyline in enumerate(polylines):
            cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'{area_names[i]}', tuple(polyline[0]), 1, 1)

        cv2.imshow('FRAME', frame)
        cv2.setMouseCallback('FRAME', draw_parking_mapping)

        key = cv2.waitKey(100) & 0xFF

        if key == ord('s') or key == ord('S'):
            data = {
                'area_id': area_id,
                'area_name': area_name,
                'source': source,
                'file_name': file_name,
                'file_path': file_path,
                'area_names': area_names,
                'polylines': [polyline.tolist() for polyline in polylines],
            }
            save_data(data)
            modal_active = True
            modal_alert("success", f"{area_name} has been saved")
            modal_active = False
        elif key == 27:
            data = get_data(file_path)
            if data:
                if area_names != data['area_names']:
                    modal_active = True
                    decision = modal_okcancel("Last changes unsaved. Discard changes?")
                    modal_active = False
                    if decision == "yes":
                        break
                else:
                    break
            else:
                modal_active = True
                decision = modal_okcancel("Data unsaved. Leave?")
                modal_active = False
                if decision == "yes":
                    break

    video.release()
    cv2.destroyAllWindows()
    create_data()
    app.deiconify()
    
def setup_parking_update(area_name, source, old_file_path):
    play_video = source
    if source.isdigit():
        play_video = int(source)
        
    video = cv2.VideoCapture(play_video)  # video parkir
    
    if not video.isOpened():
        modal_alert("failed", "Failed to start video")
        return

    app.withdraw()
    
    try:
        data = get_data(old_file_path)
        area_id = data["area_id"]
        area_names = data["area_names"]
        polylines = [np.array(polyline, np.int32) for polyline in data['polylines']]
    except FileNotFoundError:
        area_id = str(uuid.uuid4())
        area_names = []
        polylines = []
    
    file_name = f'{area_name.replace(" ", "_")}.json'
    file_path = f"data/parking_areas/{file_name}"
    
    coordinates = []
    drawing = False
    modal_active = False

    def draw_parking_mapping(event, x, y, flags, param):
        nonlocal polylines, area_names, coordinates, drawing, modal_active
        if modal_active:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            coordinates = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                coordinates.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if coordinates:
                modal_active = True
                current_name = modal_input("Area name")
                modal_active = False
                if current_name:
                    current_name = current_name.upper()
                    if current_name in all_area_names() or current_name in area_names:
                        modal_active = True
                        modal_alert("warning", f"{current_name} already exist")
                        modal_active = False
                    else:
                        area_names.append(current_name)
                        polylines.append(np.array(coordinates, np.int32))
                coordinates = []
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, polyline in enumerate(polylines):
                polyline_target = cv2.pointPolygonTest(polyline, (x, y), False)
                if polyline_target >= 0:
                    del polylines[i]
                    del area_names[i]
                    break

    while True:
        ret, frame = video.read()
        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (720, 480))

        for i, polyline in enumerate(polylines):
            cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'{area_names[i]}', tuple(polyline[0]), 1, 1)

        cv2.imshow('FRAME', frame)
        cv2.setMouseCallback('FRAME', draw_parking_mapping)

        key = cv2.waitKey(100) & 0xFF

        if key == ord('s') or key == ord('S'):
            os.remove(old_file_path)
            old_file_path = file_path
            data = {
                'area_id': area_id,
                'area_name': area_name,
                'source': source,
                'file_name': file_name,
                'file_path': file_path,
                'area_names': area_names,
                'polylines': [polyline.tolist() for polyline in polylines],
            }
            save_data(data)
            modal_active = True
            modal_alert("success", "All changes have been saved")
            modal_active = False
        elif key == 27:
            data = get_data(file_path)
            if data:
                if area_names != data['area_names']:
                    modal_active = True
                    decision = modal_okcancel("Last changes unsaved. Discard changes?")
                    modal_active = False
                    if decision == "yes":
                        break
                else:
                    break
            else:
                modal_active = True
                decision = modal_okcancel("Data unsaved. Leave?")
                modal_active = False
                if decision == "yes":
                    break
    data_update = [{
        "area_name":area_name,
        "source":source,
        "file_name":file_name,
        "file_path":file_path
    }]

    video.release()
    cv2.destroyAllWindows()
    update_data(data_update)
    app.deiconify()

def request_api(url, payload):
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print(f"Database diperbarui : {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Gagal memperbarui database: {e}")

def detection_parking(file_path, stop_event):
    data = get_data(file_path)
    if not data:
        return
    
    polylines = [np.array(polyline, np.int32) for polyline in data['polylines']]
    area_names = data['area_names']
    source = data['source']

    with open("model/coco.txt", "r") as my_file:
        daftar_kelas = my_file.read().split("\n")

    model = YOLO('model/yolov8s.pt')

    play_video = source
    if source.isdigit():
        play_video = int(source)
    video = cv2.VideoCapture(play_video)
    
    if not video.isOpened():
        modal_alert("failed", "Failed to start video")
        return

    while not stop_event.is_set():
        ret, frame = video.read()
        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (720, 480))
        hasil = model.predict(frame)
        a = hasil[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        list2 = []
        car = []
        entry_field = []
        empty_space = []
        parking_area = []

        for index, row in px.iterrows():
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            d = int(row[5])
            c = daftar_kelas[d]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if 'car' in c:
                list2.append([cx, cy])

        for i, polyline in enumerate(polylines):
            parking_area.append(i)
            for cx1, cy1 in list2:
                hasil = cv2.pointPolygonTest(polyline, (cx1, cy1), False)
                if hasil >= 0:
                    car.append(cx1)
                    entry_field.append(area_names[i])
                    empty_field = [x for x in area_names if x not in entry_field]

        datas = f"Parking : {len(parking_area) - len(car)}/{len(parking_area)}\nFilled Space : {len(entry_field)} : {entry_field}\nEmpty Space : {len(empty_field)} : {empty_field}"
        print(datas)
        
        send_datas = []
        for data in area_names:
            send_datas.append({
                "area":data,
                "status": True if data in entry_field else False
            })
            
        print(send_datas)
        request_api("https://harezayoankristianto.online/api/areas", send_datas)

    video.release()
    cv2.destroyAllWindows()

def start_detection_parking():
    select_rows = select_multi_row()
    parkings = []
    area_names = []
    send_datas = []
    
    if not select_rows:
        return
    
    for select_row in select_rows:
        parkings.append(get_data(select_row["file_path"]))
        area_names.extend(get_data(select_row["file_path"])["area_names"])
    
    for data in area_names:
        send_datas.append({
            "area":data,
            "status": True
        })
    
    request_api("https://harezayoankristianto.online/api/mainareas", parkings)
    
    # Membuat stop event
    stop_event = threading.Event()

    # Membuat dan memulai thread untuk setiap set data
    threads = []
    for data in parkings:
        thread = threading.Thread(target=detection_parking, args=(data['file_path'], stop_event))
        thread.start()
        threads.append(thread)

    def close(send_datas):
        stop_event.set()
        for thread in threads:
            thread.join()
        request_api("https://harezayoankristianto.online/api/areas", send_datas)
        print("Program dihentikan")
        control_window.destroy()
    
    control_window = ctk.CTkToplevel(app)
    control_window.title("")
    control_window.geometry(geometry_center(300, 180))
    control_window.resizable(False, False)
    
    frame_header = ctk.CTkFrame(control_window, bg_color="#18181b", fg_color="#18181b")
    frame_header.pack(fill="x")
    
    frame_msg = ctk.CTkFrame(control_window, bg_color="#27272a", fg_color="#27272a")
    frame_msg.pack(expand="true", fill="both")
    
    frame_btn = ctk.CTkFrame(control_window, bg_color="#18181b", fg_color="#18181b")
    frame_btn.pack(fill="x")

    # Memuat dan mengubah ukuran gambar menggunakan PIL
    image_path = "images/icons/success.png"
    image = Image.open(image_path)
    resized_image = image.resize((20, 20))
    
    # Menggunakan CTkImage untuk memastikan kompatibilitas HighDPI
    image_ctk = ctk.CTkImage(light_image=resized_image, size=(20, 20))

    image_header = ctk.CTkLabel(frame_header, image=image_ctk, text="")
    image_header.pack(side="left", fill="x", padx=(20,0), pady=(20))
    
    header = ctk.CTkLabel(frame_header, text="Detection", font=("Helvetica", 15, "bold"))
    header.pack(side="left", fill="x", padx=10, pady=20) 
    
    msg = ctk.CTkLabel(frame_msg, text="Detection is running...", font=("Helvetica", 13), wraplength=300)
    msg.pack(expand="true", fill="both", padx=20)
    
    control_window.protocol("WM_DELETE_WINDOW", lambda: close(send_datas))
    
    btn = ctk.CTkButton(frame_btn, text="Stop", width=60, fg_color="#f87171", text_color="#000", hover_color="#dc2626",command=lambda: close(send_datas))
    btn.pack(side="right", padx=(10), pady=10)
    
    control_window.transient(app)
    control_window.grab_set()
    control_window.focus()
    app.wait_window(control_window)
    
def check_parking(file_path):
    print(file_path)
    data = get_data(file_path)
    if not data:
        return
    
    polylines = [np.array(polyline, np.int32) for polyline in data['polylines']]
    area_names = data['area_names']
    source = data['source']

    with open("model/coco.txt", "r") as my_file:
        daftar_kelas = my_file.read().split("\n")

    model = YOLO('model/yolov8s.pt')

    play_video = source
    if source.isdigit():
        play_video = int(source)
    video = cv2.VideoCapture(play_video)
    
    if not video.isOpened():
        modal_alert("failed", "Failed to start video")
        return

    app.withdraw()

    while True:
        ret, frame = video.read()
        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (720, 480))
        hasil = model.predict(frame)
        a = hasil[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        list2 = []
        car = []
        filled_space = []
        empty_space = []
        parking_area = []

        for index, row in px.iterrows():
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            d = int(row[5])
            c = daftar_kelas[d]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if 'car' in c:
                list2.append([cx, cy])

        for i, polyline in enumerate(polylines):
            parking_area.append(i)
            cv2.polylines(frame, [polyline], True, (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{area_names[i]}', tuple(polyline[0]), 1, 1)

            for i1 in list2:
                cx1 = i1[0]
                cy1 = i1[1]
                hasil = cv2.pointPolygonTest(polyline, ((cx1, cy1)), False)
                if hasil >= 0:
                    cv2.circle(frame, (cx1, cy1), 5, (255, 0, 0), 1)
                    cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
                    car.append(cx1)
                    filled_space.append(area_names[i])
                    empty_space = [x for x in area_names if x not in filled_space]

        empty_total = len(parking_area) - len(car)
        cvzone.putTextRect(frame, f'TEMPAT KOSONG : {empty_total}/{len(parking_area)}', (20, 50), 1, 1)
        cvzone.putTextRect(frame, f'TEMPAT TERISI : {filled_space}', (20, 100), 1, 1)
        cvzone.putTextRect(frame, f'TEMPAT KOSONG : {empty_space}', (20, 150), 1, 1)

        cv2.imshow('FRAME', frame)
        key = cv2.waitKey(100) & 0xFF
        
        if key == 27:
            break

    video.release()
    cv2.destroyAllWindows()
    app.deiconify()
    
def start_check_parking():
    select_rows = select_multi_row()
    
    if not select_rows:
        return
    
    if len(select_rows) > 1:
        modal_alert("warning", "Can only select one item")
        return
    
    file_path = select_rows[0]["file_path"]

    check_parking(file_path)

def tabel():
    global tree
    
    clear_widget(frame_content)
    clear_widget(frame_action)
    
    buttons = ["Create", "Update", "Delete", "Check", "Start"]
    
    btn = ctk.CTkButton(frame_action, text=buttons[0], width=0, fg_color="#27272a",hover_color="#16a34a", border_width=1, border_color="#4ade80", command=lambda: button_handle_click(buttons[0]))
    btn.pack(side="left", expand="true", fill="x", padx=(10,0), pady=(10))
    
    for i in range(1, 4):
        btn = ctk.CTkButton(frame_action, text=buttons[i], width=0, fg_color="#27272a",hover_color="#16a34a", border_width=1, border_color="#4ade80", command=lambda b=buttons[i]: button_handle_click(b))
        btn.pack(side="left", expand="true", fill="x", padx=(10, 0), pady=(10))
        
    btn = ctk.CTkButton(frame_action, text=buttons[4], width=0, fg_color="#4ade80", text_color="#000",hover_color="#16a34a", command=lambda: button_handle_click(buttons[4]))
    btn.pack(side="left", expand="true", fill="x", padx=(10), pady=(10))
    
    style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"))
    
    tree = ttk.Treeview(frame_content, columns=("#", "Name", "Source", "Total Area"), show="headings")
    tree.heading("#", text="#")
    tree.heading("Name", text="Name")
    tree.heading("Source", text="Source")
    tree.heading("Total Area", text="Total Area")
    
    tree.column("#", width=30, stretch=False, anchor="center")
    tree.column("Name", width=150, stretch=True)
    tree.column("Source", width=150, stretch=True)
    tree.column("Total Area", width=50, stretch=True, anchor="center")
    
    style.configure("Treeview", rowheight=25, font=("Helvetica", 10), foreground="#1f2937")
    style.map("Treeview", background=[("selected", "#4ade80")], foreground=[("selected", "#000")])
    tree.tag_configure("evenrow", background="#d1d5db")
    tree.tag_configure("oddrow", background="#e5e7eb")

    
    datas = get_all_data()
    
    for i, data in enumerate(datas):
        index = i+1
        values = (index, data["area_name"], data["source"], len(data["area_names"]), data["file_name"], data["file_path"])
        if i % 2 == 0:
            tree.insert("", "end", values=values, tags="evenrow")
        else:
            tree.insert("", "end", values=values, tags="oddrow")
        
    tree.pack(side="top", expand="true", fill="both")

def create_data():
    clear_widget(frame_content)
    clear_widget(frame_action)
    
    area_name_label = ctk.CTkLabel(frame_content, text="Name Area", font=("Helvetica", 13))
    area_name_label.grid(row=0, column=0, padx=(40,10), pady=10, sticky="e")
    
    area_name_entry = ctk.CTkEntry(frame_content)
    area_name_entry.grid(row=0, column=1, padx=(0,40), pady=10, sticky="we")
    
    source_label = ctk.CTkLabel(frame_content, text="Source", font=("Helvetica", 13))
    source_label.grid(row=1, column=0, padx=(40,10), pady=10, sticky="e")
    
    source_entry = ctk.CTkEntry(frame_content)
    source_entry.grid(row=1, column=1, padx=(0,40), pady=10, sticky="we")
    
    def next():
        area_name = " ".join(area_name_entry.get().split()).upper()
        source = source_entry.get()
        
        if not area_name or not source:
            modal_alert("warning", "Name area and Source is required")
        else:
            if re.findall(r'[^A-Za-z0-9\s]', area_name):
                modal_alert("warning", "Name area cannot use unique charecters")
            else:
                if area_name in all_area_name():
                    modal_alert("warning", f"{area_name} already exists")
                else:
                    setup_parking(area_name, source)
    
    btn = ctk.CTkButton(frame_action, text="Next", width=100, fg_color="#4ade80", text_color="#000", hover_color="#16a34a", command=next)
    btn.pack(side="right", padx=(0, 10), pady=(10))
    
    btn = ctk.CTkButton(frame_action, text="Cancel", width=100, fg_color="#27272a", border_color="#f87171", hover_color="#dc2626", border_width=1, command=lambda: tabel())
    btn.pack(side="right", padx=(0, 10), pady=(10))
    
def update_data(data):
    if not data:
        return
    
    if len(data) > 1:
        modal_alert("warning", "Can only select one item")
        return
    
    data = data[0]
    
    clear_widget(frame_content)
    clear_widget(frame_action)
    
    area_name_label = ctk.CTkLabel(frame_content, text="Name", font=("Helvetica", 12))
    area_name_label.grid(row=0, column=0, padx=(40,10), pady=10, sticky="e")
    
    area_name_entry = ctk.CTkEntry(frame_content)
    area_name_entry.grid(row=0, column=1, padx=(0,40), pady=10, sticky="we")
    area_name_entry.insert(0, data["area_name"])
    
    source_label = ctk.CTkLabel(frame_content, text="Source", font=("Helvetica", 12))
    source_label.grid(row=1, column=0, padx=(40,10), pady=10, sticky="e")
    
    source_entry = ctk.CTkEntry(frame_content)
    source_entry.grid(row=1, column=1, padx=(0,40), pady=10, sticky="we")
    source_entry.insert(0, data["source"])
    
    def next():
        area_name = " ".join(area_name_entry.get().split()).upper()
        source = source_entry.get()
        all_area_names = all_area_name()
        
        while data["area_name"] in all_area_names:
            all_area_names.remove(data["area_name"])
        
        if not area_name or not source:
            modal_alert("warning", "Name area and Source is required")
        else:
            if re.findall(r'[^A-Za-z0-9\s]', area_name):
                modal_alert("warning", "Name cannot use unique characters")
            else:
                if area_name in all_area_names:
                    modal_alert("warning", f"{area_name} already exists")
                else:
                    setup_parking_update(area_name, source, data["file_path"])
    
    btn = ctk.CTkButton(frame_action, text="Next", width=100, fg_color="#4ade80", text_color="#000", hover_color="#16a34a", command=next)
    btn.pack(side="right", padx=(0, 10), pady=(10))
    
    btn = ctk.CTkButton(frame_action, text="Cancel", width=100, fg_color="#27272a", border_color="#f87171", hover_color="#dc2626", border_width=1, command=lambda: tabel())
    btn.pack(side="right", padx=(0, 10), pady=(10))

def delete_data():
    datas = select_multi_row()
    
    if not datas:
        return

    if modal_okcancel("Delete selected items?") == "yes":
        for data in datas:
            file_path = data["file_path"]
            if file_check(file_path):
                os.remove(file_path)
        modal_alert("success", "Data success to delete")
        tabel()

def button_handle_click(button):
    if button == "Create":
        create_data()
    elif button == "Update":
        update_data(select_multi_row())
    elif button == "Delete":
        delete_data()
    elif button == "Check":
        start_check_parking()
    elif button == "Start":
        start_detection_parking()
    else:
        print(button)
        tabel()

app = ctk.CTk()
app.title("Parking Area Detection")

ctk.set_appearance_mode("dark")

style = ttk.Style()

# frame
# frame header
frame_header = ctk.CTkFrame(app, bg_color="#09090b", fg_color="#09090b")
frame_header.pack(fill="x")
frame_header.grid_columnconfigure(0, weight=1)

header_title("Parking Area Detection")

frame_body = ctk.CTkFrame(app, bg_color="#27272a", fg_color="#27272a")
frame_body.pack(expand="true", fill="both")

frame_action = ctk.CTkFrame(frame_body, bg_color="#27272a", fg_color="#27272a")
frame_action.pack(fill="x")

frame_content = ctk.CTkFrame(frame_body, bg_color="#27272a", fg_color="#27272a")
frame_content.pack(expand="true", fill="both", padx=10, pady=10)
frame_content.grid_columnconfigure(1, weight=1)

tabel()

app.geometry(geometry_center(480,300))
app.mainloop()