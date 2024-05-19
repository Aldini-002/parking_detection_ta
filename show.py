import cv2
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone

with open('parking_setup', 'rb') as f:
  data = pickle.load(f)
  polylines, area_names = data['polylines'], data['area_names']

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

model=YOLO('yolov8s.pt')

video=cv2.VideoCapture('easy1.mp4')

while True:
  ret, frame = video.read()
  if not ret:
      video.set(cv2.CAP_PROP_POS_FRAMES, 0)
      continue

  frame=cv2.resize(frame,(1500, 735))
  frame_copy = frame.copy()
  results=model.predict(frame)
  a=results[0].boxes.data
  px=pd.DataFrame(a).astype("float")
  
  list1 = []
  cars = []
  filled_space = []
  empyt_space = []
  parking_space = []
  
  for index,row in px.iterrows():
    x1=int(row[0])
    y1=int(row[1])
    x2=int(row[2])
    y2=int(row[3])
    d=int(row[5])
    
    c=class_list[d]
    
    cx=int(x1+x2)//2
    cy=int(y1+y2)//2
    
    if 'car' in c:
        list1.append([cx, cy])
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)

  for i, polyline in enumerate(polylines):
    parking_space.append(i)
    # print(area_names[i])
    cv2.polylines(frame, [polyline], True, (0,255,0), 2)
    cvzone.putTextRect(frame, f'{area_names[i]}', tuple(polyline[0]), 1, 1)
    
    for i1 in list1:
      cx1 = i1[0]
      cy1 = i1[1]
      result = cv2.pointPolygonTest(polyline, ((cx1, cy1)), False)
      # print(result)
      if result >=0:
        cv2.circle(frame, (cx1, cy1), 5, (255,0,0), 1)
        cv2.polylines(frame, [polyline], True, (0,0,255), 2)
        cars.append(cx1)
        filled_space.append(area_names[i])
        empyt_space = [x for x in area_names if x not in filled_space]

  free_space = len(parking_space) - len(cars)
  cvzone.putTextRect(frame, f'FREE SPACE : {free_space}/{len(parking_space)}', (20,50), 1, 1)
  cvzone.putTextRect(frame, f'FILLED SPACE : {filled_space}', (20,100), 1, 1)
  cvzone.putTextRect(frame, f'EMPTY SPACE : {empyt_space}', (20,150), 1, 1)
  
  cv2.imshow('FRAME', frame)
  key = cv2.waitKey(100) & 0xFF
  
  if key == 27: break

video.release()

cv2.destroyAllWindows()

