import customtkinter as ctk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import json
import os
import cv2
import re
import uuid
import threading
import requests
from ultralytics import YOLO
import pytesseract
import time
from collections import Counter
import mimetypes
from termcolor import colored
import shutil


# modal
def modal_okcancel(body):
    result = [None]
    modal = ctk.CTkToplevel(app)
    modal.title("")
    modal.geometry(geometry_center(300, 180))
    modal.resizable(False, False)

    frame_header = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_header.pack(fill="x")

    frame_body = ctk.CTkFrame(modal, bg_color="#27272a", fg_color="#27272a")
    frame_body.pack(expand="true", fill="both")

    frame_footer = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_footer.pack(fill="x")
    frame_footer.grid_columnconfigure((0, 1), weight=1)

    # Memuat dan mengubah ukuran gambar menggunakan PIL
    image_path = os.path.join(os.getcwd(), "data", "icons", "warning.png")
    image = Image.open(image_path)
    resized_image = image.resize((20, 20))

    # Menggunakan CTkImage untuk memastikan kompatibilitas HighDPI
    image_ctk = ctk.CTkImage(light_image=resized_image, size=(20, 20))

    image_header = ctk.CTkLabel(frame_header, image=image_ctk, text="")
    image_header.pack(side="left", fill="x", padx=(20, 0), pady=(20))

    header = ctk.CTkLabel(frame_header, text="Warning", font=("Helvetica", 15, "bold"))
    header.pack(side="left", fill="x", padx=10, pady=20)

    msg = ctk.CTkLabel(frame_body, text=body, font=("Helvetica", 15), anchor="w")
    msg.pack(expand="true", fill="both", padx=40)

    def close(value):
        result[0] = value
        modal.destroy()

    modal.protocol("WM_DELETE_WINDOW", lambda: close(None))

    btn = ctk.CTkButton(
        frame_footer,
        text="Cancel",
        width=60,
        fg_color="#4ade80",
        text_color="#000",
        hover_color="#16a34a",
        command=lambda: close(None),
    )
    btn.pack(side="right", padx=(10), pady=10)

    btn = ctk.CTkButton(
        frame_footer,
        text="Ok",
        width=60,
        fg_color="#27272a",
        border_width=1,
        border_color="#f87171",
        hover_color="#dc2626",
        command=lambda: close("yes"),
    )
    btn.pack(side="right", padx=0, pady=10)

    modal.transient(app)
    modal.grab_set()
    modal.focus()
    app.wait_window(modal)

    return result[0]


def modal_input(body):
    result = [None]

    modal = ctk.CTkToplevel(app)
    modal.title("")
    modal.geometry(geometry_center(300, 180))
    modal.resizable(False, False)

    frame_header = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_header.pack(fill="x")

    frame_body = ctk.CTkFrame(modal, bg_color="#27272a", fg_color="#27272a")
    frame_body.pack(expand="true", fill="both")
    frame_body.grid_columnconfigure(1, weight=1)
    frame_body.grid_rowconfigure(0, weight=1)

    frame_footer = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_footer.pack(fill="x")
    frame_footer.grid_columnconfigure((0, 1), weight=1)

    # Memuat dan mengubah ukuran gambar menggunakan PIL
    image_path = os.path.join(os.getcwd(), "data", "icons", "form.png")
    image = Image.open(image_path)
    resized_image = image.resize((20, 20))

    # Menggunakan CTkImage untuk memastikan kompatibilitas HighDPI
    image_ctk = ctk.CTkImage(light_image=resized_image, size=(20, 20))

    image_header = ctk.CTkLabel(frame_header, image=image_ctk, text="")
    image_header.pack(side="left", fill="x", padx=(20, 0), pady=(20))

    header = ctk.CTkLabel(frame_header, text="Form", font=("Helvetica", 15, "bold"))
    header.pack(side="left", fill="x", padx=10, pady=20)

    input_label = ctk.CTkLabel(frame_body, text=body, font=("Helvetica", 12))
    input_label.grid(row=0, column=0, padx=(30, 5), sticky="e")

    input_entry = ctk.CTkEntry(frame_body)
    input_entry.grid(row=0, column=1, padx=(0, 30), sticky="we")
    modal.after(100, input_entry.focus_set)

    def close(value):
        result[0] = value
        modal.destroy()

    modal.protocol("WM_DELETE_WINDOW", lambda: close(None))

    btn = ctk.CTkButton(
        frame_footer,
        text="Ok",
        width=60,
        fg_color="#4ade80",
        text_color="#000",
        hover_color="#16a34a",
        command=lambda: close(input_entry.get()),
    )
    btn.pack(side="right", padx=10, pady=10)

    btn = ctk.CTkButton(
        frame_footer,
        text="Cancel",
        width=60,
        fg_color="#27272a",
        border_width=1,
        border_color="#f87171",
        hover_color="#dc2626",
        command=lambda: close(None),
    )
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

    frame_body = ctk.CTkFrame(modal, bg_color="#27272a", fg_color="#27272a")
    frame_body.pack(expand="true", fill="both")

    frame_footer = ctk.CTkFrame(modal, bg_color="#18181b", fg_color="#18181b")
    frame_footer.pack(fill="x")

    # Memuat dan mengubah ukuran gambar menggunakan PIL
    image_path = os.path.join(os.getcwd(), "data", "icons", "warning.png")
    if status == "success":
        image_path = os.path.join(os.getcwd(), "data", "icons", "success.png")
    elif status == "failed":
        image_path = os.path.join(os.getcwd(), "data", "icons", "failed.png")
    image = Image.open(image_path)
    resized_image = image.resize((20, 20))

    # Menggunakan CTkImage untuk memastikan kompatibilitas HighDPI
    image_ctk = ctk.CTkImage(light_image=resized_image, size=(20, 20))

    image_header = ctk.CTkLabel(frame_header, image=image_ctk, text="")
    image_header.pack(side="left", fill="x", padx=(20, 0), pady=(20))

    header = ctk.CTkLabel(
        frame_header, text=status.title(), font=("Helvetica", 15, "bold")
    )
    header.pack(side="left", fill="x", padx=10, pady=20)

    msg = ctk.CTkLabel(frame_body, text=message, font=("Helvetica", 13), wraplength=300)
    msg.pack(expand="true", fill="both", padx=20)

    def close():
        modal.destroy()

    btn = ctk.CTkButton(
        frame_footer,
        text="Ok",
        width=60,
        fg_color="#4ade80",
        text_color="#000",
        hover_color="#16a34a",
        command=lambda: close(),
    )
    btn.pack(side="right", padx=(10), pady=10)

    modal.transient(app)
    modal.grab_set()
    modal.focus()
    app.wait_window(modal)


# file management
paths = {
    "parkings_path": os.path.join(os.getcwd(), "data", "json", "parkings"),
    "gates_path": os.path.join(os.getcwd(), "data", "json", "gates"),
    "img_gates_path": os.path.join(os.getcwd(), "data", "img", "gates"),
    "img_parkings_path": os.path.join(os.getcwd(), "data", "img", "parkings"),
    "videos_path": os.path.join(os.getcwd(), "data", "videos"),
    "model_path": os.path.join(os.getcwd(), "data", "model"),
    "tesseract_path": os.path.join(os.getcwd(), "data", "tesseract"),
}


def folder_create(paths):
    for key, path in paths.items():
        if not os.path.exists(path):
            print(colored(f"Folder not found : {path}", "red"))
            try:
                os.makedirs(path)
                print(colored(f"Folder created : {path}", "green"))
            except Exception as e:
                print(colored(f"Folder failed to create : {e}", "red"))


def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(colored(f"Deleted folder and all its contents: {folder_path}", "green"))
    except Exception as e:
        print(colored(f"Failed to delete folder {folder_path}: {e}", "red"))


def save_json(data):
    with open(data["file_path"], "w") as json_file:
        json.dump(data, json_file, indent=4)


def get_file(file_path):
    read_mode = "rb"
    if file_path.endswith(".json"):
        read_mode = "r"

    with open(file_path, read_mode) as file:
        if file_path.endswith(".json"):
            return json.load(file)
        else:
            return file.read()


def get_all_json(folder_path):
    json_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            json_file = get_file(file_path)
            json_files.append(json_file)
    return json_files


def get_all_parking_names():
    names = []
    file_path = paths["parkings_path"]
    parkings = get_all_json(file_path)

    for parking in parkings:
        names.append(parking["name"])

    return names


def get_all_space_names():
    space_names = []
    file_path = paths["parkings_path"]
    parkings = get_all_json(file_path)

    for parking in parkings:
        if "spaces" in parking:
            for space in parking["spaces"]:
                space_names.append(space["space_name"])

    return space_names


def get_mime_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


delete_folder(os.path.join(os.getcwd(), "data", "img"))
folder_create(paths)


# table management
def select_rows():
    row_ids = []
    datas = []

    select_rows = tree.selection()

    if select_rows:
        for row_id in select_rows:
            row_ids.append(row_id)
        for row_id in row_ids:
            row = tree.item(row_id)
            values = row["values"]
            if values:
                i, name, source, location, total_area, file_path = values
                datas.append(
                    {
                        "name": name,
                        "source": source,
                        "file_path": file_path,
                        "location": location,
                    }
                )
            else:
                modal_alert("warning", "Values is null")
    else:
        modal_alert("warning", "No data selected")
    return datas


# api management
def request_post(url, payload, success, failed):
    try:
        # response = requests.post(
        #     f"https://harezayoankristianto.online/api{url}", json=payload
        # )
        # response = requests.post(f"https://harezayoankristianto.online/api{url}", json=payload, timeout=10)
        response = requests.post(f"http://localhost:5000/api{url}", json=payload, timeout=10)
        response.raise_for_status()
        print(colored(f"{success} : {response.status_code}", "green"))
    except requests.exceptions.RequestException as e:
        print(colored(f"{failed} : {e}", "red"))


def request_put(url, payload, success, failed):
    try:
        # response = requests.put(
        #     f"https://harezayoankristianto.online/api{url}", files=payload
        # )
        # response = requests.put(f"https://harezayoankristianto.online/api{url}", files=payload, timeout=10)
        response = requests.put(f"http://localhost:5000/api{url}", files=payload, timeout=10)
        response.raise_for_status()
        print(colored(f"{success} : {response.status_code}", "green"))
    except requests.exceptions.RequestException as e:
        print(colored(f"{failed} : {e}", "red"))


# parkings management
def setup_parkings(name, old_name, source, location, case):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        modal_alert("failed", "Could not start video")
        return

    old_file_path = os.path.join(
        paths["parkings_path"], f'{old_name.replace(" ", "_")}.json'
    )

    id = str(uuid.uuid4())
    file_name = f'{name.replace(" ", "_")}.json'
    file_path = os.path.join(paths["parkings_path"], file_name)
    img = f'{name.replace(" ", "_")}.png'
    img_path = os.path.join(paths["img_parkings_path"], img)

    points = np.array([], dtype=np.int32)
    drawing = False
    modal_active = False
    spaces = []
    all_space_names = get_all_space_names()

    if case == "update":
        try:
            parkings = get_file(old_file_path)
            id = parkings["id"]
            spaces = parkings["spaces"]
        except Exception as e:
            modal_alert("failed", "Could not load data")
            return

    app.withdraw()

    def mouse_event(event, x, y, flags, params):
        nonlocal points, drawing, spaces, modal_active, all_space_names
        if modal_active:
            return

        space_names_in_spaces = {space["space_name"] for space in spaces}
        all_space_names = [
            item for item in all_space_names if item not in space_names_in_spaces
        ]

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            points = np.array([[x, y]], dtype=np.int32)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                coordinate = np.array([[x, y]], dtype=np.int32)
                points = np.vstack([points, coordinate])
        elif event == cv2.EVENT_LBUTTONUP:
            if drawing:
                drawing = False
                modal_active = True
                current_name = modal_input("Space name")
                if current_name:
                    current_name = current_name.upper()
                    if (
                        current_name in all_space_names
                        or current_name in space_names_in_spaces
                    ):
                        modal_alert("warning", f"{current_name} already exist")
                    else:
                        spaces.append(
                            {"space_name": current_name, "polylines": points.tolist()}
                        )
                modal_active = False
                points = np.array([], dtype=np.int32)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if not drawing:
                for space in spaces:
                    target = cv2.pointPolygonTest(
                        np.array([space["polylines"]], dtype=np.int32), (x, y), False
                    )
                    if target >= 0:
                        spaces = [
                            item
                            for item in spaces
                            if item["space_name"] != space["space_name"]
                        ]

    model = YOLO(os.path.join(paths["model_path"], "yolov8s.pt"))

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (720, 480))
        # frame = cv2.flip(frame, 1)

        results = model(frame)
        cars = []
        space_names = [space["space_name"] for space in spaces]
        empty_space = space_names
        filled_space = []

        if not drawing:
            for result in results:
                for box in result.boxes:
                    cx, cy, w, h = box.xywh[0].cpu().numpy().tolist()
                    cx, cy, w, h = int(cx), int(cy), int(w), int(h)
                    confidence = box.conf.cpu().item()
                    class_id = box.cls.cpu().item()
                    class_name = model.names[class_id]

                    if class_name == "car" or class_name == "truck":
                        cars.append([cx, cy])

        for space in spaces:
            polylines = np.array([space["polylines"]], dtype=np.int32)
            cv2.polylines(
                frame, [polylines], isClosed=True, color=(255, 255, 0), thickness=2
            )

            centroid = np.mean(polylines[0], axis=0).astype(int)
            centroid = (centroid[0], centroid[1])

            text = space["space_name"]
            font_scale = 0.5
            font_thickness = 1
            text_size, _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            text_x = centroid[0] - text_size[0] // 2
            text_y = centroid[1] + text_size[1] // 2

            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                font_thickness,
                cv2.LINE_AA,
            )

            for car in cars:
                cx = car[0]
                cy = car[1]

                target = cv2.pointPolygonTest(polylines, (cx, cy), False)
                if target >= 0:
                    cv2.polylines(
                        frame,
                        [polylines],
                        isClosed=True,
                        color=(255, 0, 255),
                        thickness=2,
                    )
                    filled_space.append(space["space_name"])
                    empty_space = [
                        item for item in space_names if item not in filled_space
                    ]

        if drawing and points.size > 0:
            cv2.polylines(
                frame, [points], isClosed=True, color=(255, 255, 0), thickness=2
            )

        cv2.putText(
            frame,
            f"Parking spaces : {space_names}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Empty spaces : {empty_space}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Filled spaces : {filled_space}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

        cv2.imshow(name, frame)
        cv2.setMouseCallback(name, mouse_event)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            modal_active = True
            decision = modal_okcancel(f"Cancel to {case}")
            modal_active = False
            if decision == "yes":
                break
        elif key == ord("s"):
            if len(spaces):
                if case == "update":
                    try:
                        os.remove(old_file_path)
                    except Exception as e:
                        print(colored(f"Delete old data failed : {e}", "red"))
                data = {
                    "id": id,
                    "name": name,
                    "source": source,
                    "file_name": file_name,
                    "file_path": file_path,
                    "img": img,
                    "img_path": img_path,
                    "location": location,
                    "spaces": spaces,
                }
                save_json(data)
                modal_active = True
                modal_alert("success", f"Success to {case}")
                modal_active = False
                break
            else:
                modal_active = True
                modal_alert("failed", f"Space name cannot be empty")
                modal_active = False

    cap.release()
    cv2.destroyAllWindows()
    if case == "update":
        updated_parkings = [{"name": name, "source": source, "location": location}]
        form_parkings_page("update", updated_parkings)
    else:
        form_parkings_page("add", "")
    app.deiconify()


def parking_detections(file_path, stop_event):
    try:
        parking = get_file(file_path)
        id = parking["id"]
        name = parking["name"]
        source = parking["source"]
        img = parking["img"]
        img_path = parking["img_path"]
        spaces = parking["spaces"]
        location = parking["location"]
    except Exception as e:
        modal_alert("failed", "Could not load data")
        return

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        modal_alert("failed", "Could not open video")
        return ()

    model = YOLO(os.path.join(paths["model_path"], "yolov8s.pt"))

    old_space_status = []

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (720, 480))

        results = model(frame)

        cars = []
        space_names = [space["space_name"] for space in spaces]
        filled_space = []
        for result in results:
            for box in result.boxes:
                cx, cy, w, h = box.xywh[0].cpu().numpy().tolist()
                confidence = box.conf.cpu().item()
                class_id = box.cls.cpu().item()
                class_name = model.names[class_id]

                if class_name == "car" or class_name == "truck":
                    cars.append([cx, cy])

        for space in spaces:
            polylines = np.array([space["polylines"]], dtype=np.int32)
            cv2.polylines(
                frame, [polylines], isClosed=True, color=(255, 255, 0), thickness=2
            )

            centroid = np.mean(polylines[0], axis=0).astype(int)
            centroid = (centroid[0], centroid[1])

            text = space["space_name"]
            font_scale = 0.5
            font_thickness = 1
            text_size, _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            text_x = centroid[0] - text_size[0] // 2
            text_y = centroid[1] + text_size[1] // 2

            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                font_thickness,
                cv2.LINE_AA,
            )

            for car in cars:
                cx = int(car[0])
                cy = int(car[1])

                target = cv2.pointPolygonTest(polylines, (cx, cy), False)
                if target >= 0:
                    cv2.polylines(
                        frame,
                        [polylines],
                        isClosed=True,
                        color=(255, 0, 255),
                        thickness=2,
                    )
                    filled_space.append(space["space_name"])

        spaces_status = [
            {"space_name": space_name, "status": space_name in filled_space}
            for space_name in space_names
        ]

        cv2.putText(
            frame, f"{name}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        cv2.putText(
            frame,
            f"Location : {location}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            1,
        )

        if old_space_status != spaces_status:
            old_space_status = spaces_status
            request_post(
                "/parking_spaces",
                spaces_status,
                f"\t{name} : Success to update",
                f"\t{name} : Failed to update",
            )

        cv2.imwrite(img_path, frame)
        files = {"image": (img, get_file(img_path), get_mime_type(img_path))}
        request_put(
            f"/parking_images/{id}",
            files,
            f"\t{img} : Success to update",
            f"\t{img} : Failed to update",
        )

    try:
        os.remove(img_path)
    except Exception as e:
        print(colored(f"Delete image failed : {e}", "red"))
    cap.release()
    cv2.destroyAllWindows()


# gates management
pytesseract.pytesseract.tesseract_cmd = os.path.join(
    paths["tesseract_path"], "tesseract.exe"
)


def setup_gates(source, case):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        modal_alert("failed", "Could not open video")
        return

    name = case.upper()
    source = source
    file_name = f'{name.replace(" ", "_")}.json'
    file_path = os.path.join(paths["gates_path"], file_name)
    img = f'{name.replace(" ", "_")}.png'
    img_path = os.path.join(paths["img_gates_path"], img)

    polylines = np.array([], dtype=np.int32).reshape(-1, 2)
    drawing = False
    modal_active = False

    try:
        gate = get_file(file_path)
        polylines = np.array([gate["polylines"]], dtype=np.int32).reshape(-1, 2)
    except Exception as e:
        print(colored(f"Error : {e}", "red"))

    app.withdraw()

    def mouse_event(event, x, y, flags, params):
        nonlocal drawing, polylines, modal_active

        if modal_active:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if polylines.size > 0:
                modal_active = True
                modal_alert("warning", "Polyline has already been created")
                modal_active = False
                return
            drawing = True
            polylines = np.array([[x, y]], dtype=np.int32)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                coordinate = np.array([[x, y]], dtype=np.int32)
                polylines = np.vstack([polylines, coordinate])
        elif event == cv2.EVENT_LBUTTONUP:
            if drawing:
                drawing = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            if not drawing:
                target = cv2.pointPolygonTest(polylines, (x, y), False)
                if target >= 0:
                    polylines = np.array([], dtype=np.int32).reshape(-1, 2)

    plate = []
    model = YOLO(os.path.join(paths["model_path"], "best.pt"))

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (720, 480))

        results = model(frame)

        bounding_box = []
        if not drawing:
            for result in results:
                for box in result.boxes:
                    cx, cy, w, h = box.xywh[0].cpu().numpy().tolist()
                    cx, cy, w, h = int(cx), int(cy), int(w), int(h)
                    confidence = box.conf.cpu().item()
                    class_id = box.cls.cpu().item()
                    class_name = model.names[class_id]

                    x1 = int(cx - w / 2)
                    y1 = int(cy - h / 2)
                    x2 = int(cx + w / 2)
                    y2 = int(cy + h / 2)

                    if class_name == "numberplate":
                        bounding_box = [cx, cy, x1, y1, x2, y2]

        if polylines.size > 0:
            cv2.polylines(
                frame, [polylines], isClosed=True, color=(255, 0, 255), thickness=2
            )

            if bounding_box:
                cx, cy, x1, y1, x2, y2 = bounding_box
                target = cv2.pointPolygonTest(polylines, (cx, cy), False)
                if target >= 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                    roi = frame[y1:y2, x1:x2]
                    text = pytesseract.image_to_string(roi, config="--psm 6").strip()

                    text = re.sub(r"[^A-Z0-9]", "", text).upper()

                    if text:
                        print(colored("\tDetected", "green"))
                        plate.append(text)
                        cv2.putText(
                            frame,
                            text,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )

        if drawing and polylines.size > 0:
            cv2.polylines(
                frame, [polylines], isClosed=True, color=(255, 0, 255), thickness=2
            )

        cv2.imshow(name, frame)
        cv2.setMouseCallback(name, mouse_event)

        key = cv2.waitKey(100) & 0xFF
        if key == ord("q"):
            modal_active = True
            decision = modal_okcancel(f"{name} cancel to save")
            modal_active = False
            if decision == "yes":
                break
        elif key == ord("s"):
            if polylines.size:
                data = {
                    "name": name,
                    "source": source,
                    "file_name": file_name,
                    "file_path": file_path,
                    "img": img,
                    "img_path": img_path,
                    "polylines": polylines.tolist(),
                }
                save_json(data)
                modal_active = True
                modal_alert("success", "Success to save")
                modal_active = False
                break
            else:
                modal_active = True
                modal_alert("failed", f"Polylines cannot be empty")
                modal_active = False

    cap.release()
    cv2.destroyAllWindows()
    setup_gates_page()
    app.deiconify()


def plate_detections(file_path, stop_event):
    try:
        gates = get_file(file_path)
        name = gates["name"]
        source = gates["source"]
        file_path = gates["file_path"]
        img_path = gates["img_path"]
        polylines = np.array([gates["polylines"]], dtype=np.int32).reshape(-1, 2)
    except Exception as e:
        modal_alert("failed", "Could not load data")
        return

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        modal_alert("failed", "Could not open video")
        return ()

    plate_path = os.path.join(paths["gates_path"], "plate.json")

    url_gate_in = "/parkings/in"
    url_gate_out = "/parkings/out"
    url_gate = "/gates/in"
    plate = []
    model = YOLO(os.path.join(paths["model_path"], "best.pt"))

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (720, 480))

        results = model(frame)

        bounding_box = []
        for result in results:
            for box in result.boxes:
                cx, cy, w, h = box.xywh[0].cpu().numpy().tolist()
                cx, cy, w, h = int(cx), int(cy), int(w), int(h)
                confidence = box.conf.cpu().item()
                class_id = box.cls.cpu().item()
                class_name = model.names[class_id]

                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)

                if class_name == "numberplate":
                    bounding_box = [cx, cy, x1, y1, x2, y2]

        cv2.polylines(
            frame, [polylines], isClosed=True, color=(255, 0, 255), thickness=2
        )

        if bounding_box:
            cx, cy, x1, y1, x2, y2 = bounding_box
            target = cv2.pointPolygonTest(polylines, (cx, cy), False)
            if target >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                roi = frame[y1:y2, x1:x2]
                text = pytesseract.image_to_string(roi, config="--psm 6").strip()

                text = re.sub(r"[^A-Z0-9]", "", text).upper()

                if text:
                    print(colored("\tDetected", "green"))
                    plate.append(text)
                    cv2.putText(
                        frame,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )
                    if name == "GATE OUT":
                        request_post(
                            url_gate_out,
                            {"code": text},
                            f"Gate Out : {text} : Success to send",
                            f"\tGate out : Failed to send",
                        )

        if name == "GATE IN":
            plate_counter = Counter(plate)
            for key, count in plate_counter.items():
                if count >= 5:
                    cv2.putText(
                        frame,
                        f"Plat {key} terdeteksi {count} kali",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )
                    cv2.imwrite(img_path, frame)
                    plates = []
                    try:
                        plates = get_file(plate_path)
                    except Exception as e:
                        print(colored(f"Error: {e}", "red"))

                    # if key not in plates:
                    plates.append(key)
                    request_post(
                        url_gate_in,
                        {"code": key},
                        f"\tGate In : {key} : Success to send",
                        f"\tGate In : Failed to send",
                    )
                    request_post(
                        url_gate,
                        {"gateStatus": True},
                        f"\tSuccess to open gate",
                        f"\tFailed to open gate",
                    )
                    with open(plate_path, "w") as json_file:
                        json.dump(plates, json_file, indent=4)
                    cv2.imwrite(img_path, frame)
                    cv2.waitKey(100)
                    time.sleep(3)
                    request_post(
                        url_gate,
                        {"gateStatus": False},
                        f"\tSuccess to close gate",
                        f"\tFailed to close gate",
                    )
                    plate = []
        else:
            plates = []
            try:
                plates = get_file(plate_path)
            except FileNotFoundError:
                None

            if plates:
                plates = [item for item in plates if item not in plate]

                cv2.putText(
                    frame,
                    f"{plates}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

                with open(plate_path, "w") as json_file:
                    json.dump(plates, json_file, indent=4)
            plate = []

        # cv2.imshow(name, frame)
        cv2.imwrite(img_path, frame)

    try:
        os.remove(img_path)
    except Exception as e:
        print(colored(f"Delete image failed : {e}", "red"))
    cap.release()
    cv2.destroyAllWindows()


# start system
def start(select_rows):
    parkings = []

    # try:
    #     gate_in = get_file(os.path.join(paths["gates_path"], "GATE_IN.json"))
    #     gate_out = get_file(os.path.join(paths["gates_path"], "GATE_OUT.json"))
    # except Exception as e:
    #     modal_alert("failed", "Please complete gate setup")
    #     setup_gates_page()
    #     return

    # if gate_in["source"] == gate_out["source"]:
    #     modal_alert("failed", "Source gate could not same")
    #     setup_gates_page()
    #     return

    if not select_rows:
        return

    for select_row in select_rows:
        parkings.append(get_file(select_row["file_path"]))

    for parking in parkings:
        space_names = [space["space_name"] for space in parking.pop("spaces")]
        parking["space_names"] = space_names

    request_post(
        "/parking_areas",
        parkings,
        f"\tSuccess to update parkings ",
        f" Failed to update parkings ",
    )

    stop_event = threading.Event()

    threads = []
    # thread_gate_in = threading.Thread(
    #     target=plate_detections, args=(gate_in["file_path"], stop_event)
    # )
    # thread_gate_out = threading.Thread(
    #     target=plate_detections, args=(gate_out["file_path"], stop_event)
    # )

    # thread_gate_in.start()
    # thread_gate_out.start()

    # threads.append(thread_gate_in)
    # threads.append(thread_gate_out)

    for parking in parkings:
        thread = threading.Thread(
            target=parking_detections, args=(parking["file_path"], stop_event)
        )
        threads.append(thread)
        thread.start()

    def close():
        stop_event.set()
        for thread in threads:
            thread.join()
        request_post(
            "/parking_areas/delete",
            None,
            f"\tSuccess to update parkings ",
            f" Failed to update parkings ",
        )
        modal_alert("success", "System is stopped")
        control_window.destroy()

    control_window = ctk.CTkToplevel(app)
    control_window.title("")
    control_window.geometry(geometry_center(980, 530))
    control_window.resizable(False, False)

    def show_parking_detections(parkings):
        parking_detections_window = ctk.CTkToplevel(app)
        parking_detections_window.title("")
        parking_detections_window.geometry(geometry_center(750, 400))
        parking_detections_window.resizable(False, False)

        frame_header = ctk.CTkFrame(
            parking_detections_window, bg_color="#18181b", fg_color="#18181b"
        )
        frame_header.pack(fill="x")

        frame_body = ctk.CTkScrollableFrame(
            parking_detections_window, bg_color="#27272a", fg_color="#27272a"
        )
        frame_body.pack(expand="true", fill="both")
        frame_body.grid_columnconfigure((0, 1), weight=1)

        frame_footer = ctk.CTkFrame(
            parking_detections_window, bg_color="#18181b", fg_color="#18181b"
        )
        frame_footer.pack(fill="x")

        # Memuat dan mengubah ukuran gambar menggunakan PIL
        icon_path = os.path.join(os.getcwd(), "data", "icons", "warning.png")
        icon = Image.open(icon_path)
        resized_icon = icon.resize((20, 20))

        # Menggunakan CTkImage untuk memastikan kompatibilitas HighDPI
        icon_ctk = ctk.CTkImage(light_image=resized_icon, size=(20, 20))

        image_header = ctk.CTkLabel(frame_header, image=icon_ctk, text="")
        image_header.pack(side="left", fill="x", padx=(20, 0), pady=(20))

        header = ctk.CTkLabel(
            frame_header,
            text="Parking Detections".title(),
            font=("Helvetica", 15, "bold"),
        )
        header.pack(side="left", fill="x", padx=10, pady=10)

        for i, parking in enumerate(parkings):
            row = i // 2
            column = i % 2
            padx = (10, 10)
            if column == 1:
                padx = (0, 10)

            parking_label = ctk.CTkLabel(frame_body, text="")
            parking_label.grid(row=row, column=column, padx=padx, pady=10)

            def update_video_label(label, image_path):
                try:
                    image = Image.open(image_path)
                    resized_image = image.resize((360, 240))
                    image_tk = ImageTk.PhotoImage(resized_image)
                    label.configure(image=image_tk)
                    label.image = image_tk
                except Exception as e:
                    None

            def update_videos(label, image_path):
                update_video_label(label, image_path)
                parking_detections_window.after(100, update_videos, label, image_path)

            update_videos(parking_label, parking["img_path"])

        def close():
            parking_detections_window.destroy()

        btn = ctk.CTkButton(
            frame_footer,
            text="Close",
            width=60,
            fg_color="#4ade80",
            text_color="#000",
            hover_color="#16a34a",
            command=close,
        )
        btn.pack(side="right", padx=(10), pady=10)

        parking_detections_window.lift()
        parking_detections_window.focus_force()
        parking_detections_window.grab_set()

        parking_detections_window.deiconify()

    frame_header = ctk.CTkFrame(control_window, bg_color="#18181b", fg_color="#18181b")
    frame_header.pack(fill="x")

    frame_msg = ctk.CTkFrame(control_window, bg_color="#27272a", fg_color="#27272a")
    frame_msg.pack(fill="x")
    frame_msg.grid_columnconfigure((0, 1), weight=1)

    frame_gate = ctk.CTkFrame(control_window, bg_color="#27272a", fg_color="#27272a")
    frame_gate.pack(expand="true", fill="both")
    frame_gate.grid_columnconfigure((0, 1), weight=1)

    frame_btn = ctk.CTkFrame(control_window, bg_color="#18181b", fg_color="#18181b")
    frame_btn.pack(fill="x")

    # Memuat dan mengubah ukuran gambar menggunakan PIL
    image_path = os.path.join(os.getcwd(), "data", "icons", "success.png")
    image = Image.open(image_path)
    resized_image = image.resize((20, 20))

    # Menggunakan CTkImage untuk memastikan kompatibilitas HighDPI
    image_ctk = ctk.CTkImage(light_image=resized_image, size=(20, 20))

    image_header = ctk.CTkLabel(frame_header, image=image_ctk, text="")
    image_header.pack(side="left", fill="x", padx=(20, 0), pady=(20))

    header = ctk.CTkLabel(
        frame_header, text="Smart Parking System", font=("Helvetica", 15, "bold")
    )
    header.pack(side="left", fill="x", padx=10, pady=20)

    msg = ctk.CTkLabel(
        frame_msg, text="System is running...", font=("Helvetica", 13), wraplength=300
    )
    msg.grid(row=0, column=0, columnspan=2, pady=10, sticky="we")

    gate_in_label = ctk.CTkLabel(
        frame_msg, text="GATE IN", font=("Helvetica", 13), wraplength=300
    )
    gate_in_label.grid(row=1, column=0, sticky="we")

    gate_out_label = ctk.CTkLabel(
        frame_msg, text="GATE OUT", font=("Helvetica", 13), wraplength=300
    )
    gate_out_label.grid(row=1, column=1, sticky="we")

    gate_in_video = ctk.CTkLabel(frame_gate, text="")
    gate_in_video.pack(expand="true", fill="both", side="left")

    gate_out_video = ctk.CTkLabel(frame_gate, text="")
    gate_out_video.pack(expand="true", fill="both", side="right")

    def update_video_label(label, image_path):
        try:
            image = Image.open(image_path)
            resized_image = image.resize((480, 320))
            image_tk = ImageTk.PhotoImage(resized_image)
            label.configure(image=image_tk)
            label.image = image_tk
        except Exception as e:
            None

    def update_videos():
        gate_in_image_path = os.path.join(
            os.getcwd(), "data", "img", "gates", "GATE_IN.png"
        )
        gate_out_image_path = os.path.join(
            os.getcwd(), "data", "img", "gates", "GATE_OUT.png"
        )

        update_video_label(gate_in_video, gate_in_image_path)
        update_video_label(gate_out_video, gate_out_image_path)

        control_window.after(100, update_videos)

    # Start the periodic update of the video labels
    update_videos()

    control_window.protocol("WM_DELETE_WINDOW", close)

    btn = ctk.CTkButton(
        frame_btn,
        text="Stop",
        width=60,
        fg_color="#f87171",
        text_color="#000",
        hover_color="#dc2626",
        command=close,
    )
    btn.pack(side="right", padx=(10), pady=10)

    btn_show = ctk.CTkButton(
        frame_btn,
        text="Show Parking Detections",
        width=200,
        fg_color="#60a5fa",
        text_color="#000",
        hover_color="#2563eb",
        command=lambda: show_parking_detections(parkings),
    )
    btn_show.pack(side="right", pady=10)

    control_window.transient(app)
    control_window.grab_set()
    control_window.focus()
    app.wait_window(control_window)


# ctk management
def geometry_center(app_width, app_height):
    screen_width = app.winfo_screenwidth()
    screen_height = app.winfo_screenheight()

    x = int((screen_width - app_width) / 2)
    y = int((screen_height - app_height) / 2)

    center = f"{app_width}x{app_height}+{x}+{y}"

    return center


def clear_widget(frame):
    for widget in frame.winfo_children():
        widget.destroy()


def show_tabel():
    global tree

    clear_widget(frame_content)
    clear_widget(frame_action)

    # button management
    buttons = ["Add New", "Update", "Delete", "Setup Gate", "Start"]

    btn = ctk.CTkButton(
        frame_action,
        text=buttons[0],
        width=0,
        fg_color="#27272a",
        hover_color="#16a34a",
        border_width=1,
        border_color="#4ade80",
        command=lambda: button_handle_click(buttons[0]),
    )
    btn.pack(side="left", expand="true", fill="x", padx=(10, 0), pady=(10))

    for i in range(1, 3):
        btn = ctk.CTkButton(
            frame_action,
            text=buttons[i],
            width=0,
            fg_color="#27272a",
            hover_color="#16a34a",
            border_width=1,
            border_color="#4ade80",
            command=lambda b=buttons[i]: button_handle_click(b),
        )
        btn.pack(side="left", expand="true", fill="x", padx=(10, 0), pady=(10))

    btn = ctk.CTkButton(
        frame_action,
        text=buttons[3],
        width=0,
        fg_color="#60a5fa",
        text_color="#000",
        hover_color="#2563eb",
        command=lambda: button_handle_click(buttons[3]),
    )
    btn.pack(side="left", expand="true", fill="x", padx=(10, 0), pady=(10))

    btn = ctk.CTkButton(
        frame_action,
        text=buttons[4],
        width=0,
        fg_color="#4ade80",
        text_color="#000",
        hover_color="#16a34a",
        command=lambda: button_handle_click(buttons[4]),
    )
    btn.pack(side="left", expand="true", fill="x", padx=(10), pady=(10))

    # table management
    style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"))

    tree = ttk.Treeview(
        frame_content,
        columns=("#", "Name", "Source", "Location", "Total Area"),
        show="headings",
    )
    tree.heading("#", text="#")
    tree.heading("Name", text="Name")
    tree.heading("Source", text="Source")
    tree.heading("Location", text="Location")
    tree.heading("Total Area", text="Total Area")

    tree.column("#", width=30, stretch=False, anchor="center")
    tree.column("Name", width=150, stretch=True)
    tree.column("Source", width=150, stretch=True)
    tree.column("Location", width=150, stretch=True)
    tree.column("Total Area", width=50, stretch=True, anchor="center")

    style.configure(
        "Treeview", rowheight=25, font=("Helvetica", 10), foreground="#1f2937"
    )
    style.map(
        "Treeview",
        background=[("selected", "#4ade80")],
        foreground=[("selected", "#000")],
    )
    tree.tag_configure("evenrow", background="#d1d5db")
    tree.tag_configure("oddrow", background="#e5e7eb")

    parkings = get_all_json(paths["parkings_path"])

    for i, parking in enumerate(parkings):
        index = i + 1
        values = (
            index,
            parking["name"],
            parking["source"],
            parking["location"],
            len(parking["spaces"]),
            parking["file_path"],
        )
        if i % 2 == 0:
            tree.insert("", "end", values=values, tags="evenrow")
        else:
            tree.insert("", "end", values=values, tags="oddrow")

    tree.pack(side="top", expand="true", fill="both")


def form_parkings_page(case, parking):
    if case == "update":
        if not parking:
            return

        if len(parking) > 1:
            modal_alert("warning", "Can only select one data")
            return

        parking = parking[0]

    clear_widget(frame_content)
    clear_widget(frame_action)

    name_label = ctk.CTkLabel(
        frame_content, text="Parking name", font=("Helvetica", 12)
    )
    name_label.grid(row=0, column=0, padx=(40, 10), pady=10, sticky="e")

    name_entry = ctk.CTkEntry(frame_content)
    name_entry.grid(row=0, column=1, padx=(0, 40), pady=10, sticky="we")
    name_entry.insert(0, parking["name"] if case == "update" else "")
    name_entry.focus_set()

    source_label = ctk.CTkLabel(frame_content, text="Source", font=("Helvetica", 12))
    source_label.grid(row=1, column=0, padx=(40, 10), pady=10, sticky="e")

    source_entry = ctk.CTkEntry(frame_content)
    source_entry.grid(row=1, column=1, padx=(0, 40), pady=10, sticky="we")
    source_entry.insert(0, parking["source"] if case == "update" else "")

    location_label = ctk.CTkLabel(
        frame_content, text="Location", font=("Helvetica", 12)
    )
    location_label.grid(row=2, column=0, padx=(40, 10), pady=10, sticky="e")

    location_entry = ctk.CTkEntry(frame_content)
    location_entry.grid(row=2, column=1, padx=(0, 40), pady=10, sticky="we")
    location_entry.insert(0, parking["location"] if case == "update" else "")

    def next():
        name = " ".join(name_entry.get().split()).upper()
        old_name = parking["name"] if case == "update" else name
        source = source_entry.get()
        location = location_entry.get()
        all_parking_names = [
            item for item in get_all_parking_names() if item != old_name
        ]

        if not name or not source or not location:
            modal_alert("warning", "Field could not be empyt")
        else:
            if re.findall(r"[^A-Za-z0-9\s]", name):
                modal_alert("warning", "Name cannot use unique characters")
            else:
                if name in all_parking_names:
                    modal_alert("warning", f"{name} already exists")
                else:
                    setup_parkings(name, old_name, source, location, case)

    btn = ctk.CTkButton(
        frame_action,
        text="Next",
        width=100,
        fg_color="#4ade80",
        text_color="#000",
        hover_color="#16a34a",
        command=next,
    )
    btn.pack(side="right", padx=(0, 10), pady=(10))

    btn = ctk.CTkButton(
        frame_action,
        text="Cancel",
        width=100,
        fg_color="#27272a",
        border_color="#f87171",
        hover_color="#dc2626",
        border_width=1,
        command=lambda: show_tabel(),
    )
    btn.pack(side="right", padx=(0, 10), pady=(10))


def delete_parkings(parkings):
    if not parkings:
        return

    if modal_okcancel("Delete selected data?") == "yes":
        for parking in parkings:
            file_path = parking["file_path"]
            try:
                os.remove(file_path)
            except Exception as e:
                print(colored(f"Failed to delete : {e}", "red"))
        modal_alert("success", "Data success to delete")
        show_tabel()


def setup_gates_page():
    source_in = ""
    source_out = ""

    gates = []
    try:
        gates = get_all_json(paths["gates_path"])
        for gate in gates:
            if "name" in gate:
                if gate["name"] == "GATE IN":
                    source_in = gate["source"]
                elif gate["name"] == "GATE OUT":
                    source_out = gate["source"]
    except Exception as e:
        print(colored(f"Error : {e}", "red"))

    clear_widget(frame_content)
    clear_widget(frame_action)

    # entry
    gate_in_label = ctk.CTkLabel(
        frame_content, text="Source Gate In", font=("Helvetica", 12)
    )
    gate_in_label.grid(row=0, column=0, padx=(20, 10), pady=10, sticky="e")

    gate_in_entry = ctk.CTkEntry(frame_content)
    gate_in_entry.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="we")
    gate_in_entry.insert(0, source_in)
    gate_in_entry.focus_set()

    gate_in_btn = ctk.CTkButton(
        frame_content,
        text="Setup",
        width=100,
        fg_color="#4ade80",
        text_color="#000",
        hover_color="#16a34a",
        command=lambda: next(gate_in_entry.get(), "gate in"),
    )
    gate_in_btn.grid(row=0, column=2, padx=(0, 20), pady=10, sticky="we")

    gate_out_label = ctk.CTkLabel(
        frame_content, text="Source Gate Out", font=("Helvetica", 12)
    )
    gate_out_label.grid(row=1, column=0, padx=(20, 10), pady=10, sticky="e")

    gate_out_entry = ctk.CTkEntry(frame_content)
    gate_out_entry.grid(row=1, column=1, padx=(0, 10), pady=10, sticky="we")
    gate_out_entry.insert(0, source_out)

    gate_out_btn = ctk.CTkButton(
        frame_content,
        text="Setup",
        width=100,
        fg_color="#4ade80",
        text_color="#000",
        hover_color="#16a34a",
        command=lambda: next(gate_out_entry.get(), "gate out"),
    )
    gate_out_btn.grid(row=1, column=2, padx=(0, 20), pady=10, sticky="we")

    def next(source, case):
        if not source:
            modal_alert("warning", "Source is required")
        else:
            setup_gates(source, case)

    btn = ctk.CTkButton(
        frame_action,
        text="Back",
        width=100,
        fg_color="#27272a",
        border_color="#f87171",
        hover_color="#dc2626",
        border_width=1,
        command=lambda: show_tabel(),
    )
    btn.pack(side="right", padx=(0, 10), pady=(10))


def button_handle_click(button):
    if button == "Add New":
        form_parkings_page("add", "")
    elif button == "Update":
        form_parkings_page("update", select_rows())
    elif button == "Delete":
        delete_parkings(select_rows())
    elif button == "Start":
        start(select_rows())
    elif button == "Setup Gate":
        setup_gates_page()
    else:
        show_tabel()


# main frame
app = ctk.CTk()
app.title("Parking Area Detection")

ctk.set_appearance_mode("dark")

style = ttk.Style()

frame_header = ctk.CTkFrame(app, bg_color="#09090b", fg_color="#09090b")
frame_header.pack(fill="x")
frame_header.grid_columnconfigure(0, weight=1)

header_title = ctk.CTkLabel(
    frame_header, text="Smart Parking System", font=("Helvetica", 20, "bold")
)
header_title.pack(side="top", ipady=20, fill="x")

frame_body = ctk.CTkFrame(app, bg_color="#27272a", fg_color="#27272a")
frame_body.pack(expand="true", fill="both")

frame_action = ctk.CTkFrame(frame_body, bg_color="#27272a", fg_color="#27272a")
frame_action.pack(fill="x")

frame_content = ctk.CTkFrame(frame_body, bg_color="#27272a", fg_color="#27272a")
frame_content.pack(expand="true", fill="both", padx=10, pady=10)
frame_content.grid_columnconfigure(1, weight=1)

show_tabel()

app.geometry(geometry_center(700, 500))
app.mainloop()
