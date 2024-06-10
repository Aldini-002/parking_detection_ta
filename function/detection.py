import cv2
import numpy as np
import cvzone
from ultralytics import YOLO
import pandas as pd
import requests
from function.data_management import load_data

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
