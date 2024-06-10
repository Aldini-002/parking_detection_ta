import cv2
import numpy as np
import cvzone
from ultralytics import YOLO
import pandas as pd
import requests
import json
import tkinter as tk
from tkinter import simpledialog

def save_data(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)

# Fungsi untuk memuat data dari file JSON
def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)
      
def setup():
    video = cv2.VideoCapture('video/sample.mp4')  # video parkir

    try:
        data = load_data('parking_mapping.json')
        polylines = [np.array(polyline, np.int32) for polyline in data['polylines']]
        area_names = data['area_names']
    except FileNotFoundError:
        polylines = []
        area_names = []

    coordinates = []
    current_name = ''
    drawing = False

    def draw_parking_mapping(event, x, y, flags, param):
        nonlocal polylines, area_names, coordinates, current_name, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            coordinates = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                coordinates.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            root = tk.Tk()
            root.withdraw()
            current_name = simpledialog.askstring("Input", "Enter Area Name:")
            root.destroy()
            if current_name:
                area_names.append(current_name)
                polylines.append(np.array(coordinates, np.int32))
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
        frame = cv2.resize(frame, (1500, 735))

        for i, polyline in enumerate(polylines):
            cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'{area_names[i]}', tuple(polyline[0]), 1, 1)

        cv2.imshow('FRAME', frame)
        cv2.setMouseCallback('FRAME', draw_parking_mapping)

        key = cv2.waitKey(100) & 0xFF

        if key == ord('s') or key == ord('S'):
            data = {
                'polylines': [polyline.tolist() for polyline in polylines],
                'area_names': area_names
            }
            save_data('parking_mapping.json', data)
            # with open('parking_mapping', 'wb') as f:
            #     data = {'polylines': polylines, 'area_names': area_names}
            #     pickle.dump(data, f)
            print(area_names)
        elif key == 13:
            url = "https://harezayoankristianto.online/api/areas/create"

            data_untuk_dikirim = {
                'areas': area_names,
            }

            try:
                response = requests.post(url, json=data_untuk_dikirim, timeout=10)
                response.raise_for_status()
                print(f"Data terkirim dengan kode respon: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Kesalahan saat mengirim data: {e}")
        elif key == 27:
            break

    video.release()
    cv2.destroyAllWindows()

def detection():
    data = load_data('parking_mapping.json')
    polylines = [np.array(polyline, np.int32) for polyline in data['polylines']]
    nama_area = data['area_names']

    my_file = open("coco.txt", "r")
    data = my_file.read()
    daftar_kelas = data.split("\n")

    model = YOLO('yolov8s.pt')

    video = cv2.VideoCapture('video/sample.mp4')

    while True:
        ret, frame = video.read()
        if not ret:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (1500, 735))
        frame_copy = frame.copy()
        hasil = model.predict(frame)
        a = hasil[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        daftar1 = []
        mobil = []
        tempat_terisi = []
        tempat_kosong = []
        tempat_parkir = []

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])

            c = daftar_kelas[d]

            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2

            if 'car' in c:
                daftar1.append([cx, cy])

        for i, polyline in enumerate(polylines):
            tempat_parkir.append(i)
            cv2.polylines(frame, [polyline], True, (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{nama_area[i]}', tuple(polyline[0]), 1, 1)

            for i1 in daftar1:
                cx1 = i1[0]
                cy1 = i1[1]
                hasil = cv2.pointPolygonTest(polyline, ((cx1, cy1)), False)
                if hasil >= 0:
                    cv2.circle(frame, (cx1, cy1), 5, (255, 0, 0), 1)
                    cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
                    mobil.append(cx1)
                    tempat_terisi.append(nama_area[i])
                    tempat_kosong = [x for x in nama_area if x not in tempat_terisi]

        tempat_kosong_total = len(tempat_parkir) - len(mobil)
        cvzone.putTextRect(frame, f'TEMPAT KOSONG : {tempat_kosong_total}/{len(tempat_parkir)}', (20, 50), 1, 1)
        cvzone.putTextRect(frame, f'TEMPAT TERISI : {tempat_terisi}', (20, 100), 1, 1)
        cvzone.putTextRect(frame, f'TEMPAT KOSONG : {tempat_kosong}', (20, 150), 1, 1)

        cv2.imshow('FRAME', frame)
        key = cv2.waitKey(100) & 0xFF

        url = "https://harezayoankristianto.online/api/areas/update"
        
        areas = []
        for area in nama_area:
            # Tentukan status
            status = int(area in tempat_terisi)
            
            # Tambahkan data ke array3
            areas.append({"area": area, "status": status})
    
        data_untuk_dikirim = {
            'areas': areas,
        }
        
        print(data_untuk_dikirim)
        
        try:
            response = requests.post(url, json=data_untuk_dikirim, timeout=10)
            response.raise_for_status()
            print(f"Data terkirim dengan kode respon: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Kesalahan saat mengirim data: {e}")

        if key == 27:
            break

    video.release()
    cv2.destroyAllWindows()


# GUI dengan tkinter
def create_gui():
    root = tk.Tk()
    root.title("Pilihan Kode")

    label = tk.Label(root, text="Pilih kode yang ingin dijalankan:")
    label.pack(padx=10, pady=10)

    button1 = tk.Button(root, text="Setup Parkir", command=setup)
    button1.pack(padx=10, pady=10)

    button2 = tk.Button(root, text="Deteksi parkir", command=detection)
    button2.pack(padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
