import cv2
import yaml
import numpy as np
        
sum_up = 0.0
delta_list = []
frame = cv2.imread('phong_frame_02.png') #
parking_bounding_rects = []
parking_mask = []
frame_blur = cv2.GaussianBlur(frame.copy(), (5,5), 3)
frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
with open('phong_yml_02.yml', 'r') as stream:
    parking_data = yaml.load(stream)
    
if parking_data != None:
    for park in parking_data:
        points = np.array(park['points'])
        rect = cv2.boundingRect(points)
        points_shifted = points.copy()
        points_shifted[:,0] = points[:,0] - rect[0] # shift contour to region of interest
        points_shifted[:,1] = points[:,1] - rect[1]
        
        parking_bounding_rects.append(rect)
        mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8), [points_shifted], contourIdx=-1,
                                    color=255, thickness=-1, lineType=cv2.LINE_8)
        mask = mask==255
        parking_mask.append(mask)

for ind, park in enumerate(parking_data):
        points = np.array(park['points'])
        rect = parking_bounding_rects[ind]
        roi_gray = frame_gray[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])] # crop ROI để tính toán nhanh hơn

        laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
        points[:,0] = points[:,0] - rect[0] # shift contour to roi
        points[:,1] = points[:,1] - rect[1]
        delta = np.mean(np.abs(laplacian * parking_mask[ind]))
        if(delta > 1.8):    # Bỏ qua space trống
            delta_list.append(delta)
        sum_up = sum_up + delta
        
avg = sum_up/len(parking_data)
med = statistics.median(delta_list)
print("mean: ", avg)
print("median: ", med)