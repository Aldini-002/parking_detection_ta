import cv2
import numpy as np
import cvzone
import pickle
import tkinter as tk
from tkinter import simpledialog

video = cv2.VideoCapture('easy1.mp4') #video parkir

try:
  with open('parking_setup', 'rb') as f:
    data = pickle.load(f)
    polylines, area_names = data['polylines'], data['area_names']
except:
  polylines = []
  area_names = []

coordinates = []
current_name = ''
drawing = False

# function draw parking mapping
def draw_parking_setup(event, x, y, flags, param):
    global polylines, area_names, coordinates, current_name, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        coordinates = [(x,y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:  # hanya tambahkan titik jika mouse ditekan
            coordinates.append((x,y))
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
 

# play video
while True:
  ret,frame = video.read()
  if not ret:
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    continue
  frame = cv2.resize(frame, (1500, 735))
  
  for i, polyline in enumerate(polylines):
    cv2.polylines(frame, [polyline],  True, (0,0,255), 2)
    cvzone.putTextRect(frame, f'{area_names[i]}', tuple(polyline[0]), 1, 1)
    
  cv2.imshow('FRAME', frame)
  cv2.setMouseCallback('FRAME', draw_parking_setup)
  
  key = cv2.waitKey(100) & 0xFF
  
  if key == ord('s') or key == ord('S'):
    with open('parking_setup', 'wb') as f:
      data = {'polylines':polylines, 'area_names':area_names}
      pickle.dump(data, f)
  elif key == 27:
    break
  
video.release()
cv2.destroyAllWindows()
