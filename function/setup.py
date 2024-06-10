import cv2
import numpy as np
import cvzone
# import pickle
import tkinter as tk
from tkinter import simpledialog
import requests
from function.data_management import load_data, save_data

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
