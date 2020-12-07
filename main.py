import yaml
import numpy as np
import cv2

fn = "phong_testvideo_03.mp4"  # 3
fn_yaml = "phong_yml_02.yml"
fn_out = "phong_outputvideo_02.avi"
cascade_src = 'phong_classifier.xml'
car_cascade = cv2.CascadeClassifier(cascade_src)
global_str = "Last change at: "
change_pos = 0.00
dict = {
    'print_out':True,
    'text_overlay': True,
    'parking_overlay': True,
    'parking_id_overlay': True,
    'parking_detection': True,
    'motion_detection': False,
    'pedestrian_detection': False,  # mất nhiều power
    'min_area_motion_contour': 500,
    'park_laplacian_th': 2.7,
    'park_sec_to_wait': 1,  # thời gian đợi để thay đổi trạng thái của region
    'start_frame': 0,  # Bắt đầu từ frame nào
    'show_ids': True,  # Hiển thị id cho từng lot
    'classifier_used': True,
    'save_video': False
}

# Set từ video
cap = cv2.VideoCapture(fn)
video_info = {'fps': cap.get(cv2.CAP_PROP_FPS),
              'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.6),
              'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.6),
              'fourcc': cap.get(cv2.CAP_PROP_FOURCC),
              'num_of_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}

cap.set(cv2.CAP_PROP_POS_FRAMES, dict['start_frame'])  # Nhảy đến frame được xác định trước


def run_classifier(img, id):
    cars = car_cascade.detectMultiScale(img, 1.1, 1)
    if cars == ():
        return False
    else:
        return True


# Định nghĩa codec và tạo VideoWriter object
if dict['save_video']:
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I',
                                    'D')  # các lựa chọn: ('P','I','M','1'), ('D','I','V','X'), ('M','J','P','G'), ('X','V','I','D')
    out = cv2.VideoWriter(fn_out, -1, 25.0, (video_info['width'], video_info['height']))

# Khởi tạo HOG descriptor/person detector. Mất rất nhiều power cho quá trình này.
if dict['pedestrian_detection']:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Sử dụng Background subtraction tách cảnh nền.
if dict['motion_detection']:
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)

# Đọc file yaml (parking space polygons)
with open(fn_yaml, 'r') as stream:
    parking_data = yaml.load(stream)
parking_contours = []
parking_bounding_rects = []
parking_mask = []
parking_data_motion = []
if parking_data != None:
    for park in parking_data:
        points = np.array(park['points'])
        rect = cv2.boundingRect(points)
        points_shifted = points.copy()
        points_shifted[:, 0] = points[:, 0] - rect[0]  # shift contour to region of interest
        points_shifted[:, 1] = points[:, 1] - rect[1]
        parking_contours.append(points)
        parking_bounding_rects.append(rect)
        mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8), [points_shifted], contourIdx=-1,
                                color=255, thickness=-1, lineType=cv2.LINE_8)
        mask = mask == 255
        parking_mask.append(mask)

kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # morphological kernel
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 19))
if parking_data != None:
    parking_status = [False] * len(parking_data)
    parking_buffer = [None] * len(parking_data)


def print_parkIDs(park, coor_points, frame_rev):
    moments = cv2.moments(coor_points)
    centroid = (int(moments['m10'] / moments['m00']) - 3, int(moments['m01'] / moments['m00']) + 3)
    # Gắn số vào các region được marked bằng tay
    cv2.putText(frame_rev, str(park['id']), (centroid[0] + 1, centroid[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_rev, str(park['id']), (centroid[0] - 1, centroid[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_rev, str(park['id']), (centroid[0] + 1, centroid[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_rev, str(park['id']), (centroid[0] - 1, centroid[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_rev, str(park['id']), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


while (cap.isOpened()):
    current_count = 0
    video_cur_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Vị trí hiện tại của video file tính theo giây
    video_cur_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Vị trí tính theo frame
    ret, frame_initial = cap.read()
    if ret == True:
        frame = cv2.resize(frame_initial, None, fx=0.6, fy=0.6)
    if ret == False:
        print("Video ended")
        break

    # Background Subtraction
    frame_blur = cv2.GaussianBlur(frame.copy(), (5, 5), 3)
    frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
    frame_out = frame.copy()

    # Hiển thị số frame trên góc trái video
    if dict['text_overlay']:
        str_on_frame = "%d/%d" % (video_cur_frame, video_info['num_of_frames'])
        cv2.putText(frame_out, str_on_frame, (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_out, global_str + str(round(change_pos, 2)) + 'sec', (5, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 0), 2, cv2.LINE_AA)

    # motion detection cho mọi objects
    if dict['motion_detection']:
        fgmask = fgbg.apply(frame_blur)
        bw = np.uint8(fgmask == 255) * 255
        bw = cv2.erode(bw, kernel_erode, iterations=1)
        bw = cv2.dilate(bw, kernel_dilate, iterations=1)
        (_, cnts, _) = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Áp loop cho contours
        for c in cnts:
            # Nếu contours quá nhỏ thì bỏ qua
            if cv2.contourArea(c) < dict['min_area_motion_contour']:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # Detect xe và chỗ trống
    if dict['parking_detection']:
        for ind, park in enumerate(parking_data):
            points = np.array(park['points'])
            rect = parking_bounding_rects[ind]
            roi_gray = frame_gray[rect[1]:(rect[1] + rect[3]),
                       rect[0]:(rect[0] + rect[2])]  # crop ROI để tính toán nhanh hơn

            laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
            points[:, 0] = points[:, 0] - rect[0]  # Chuyển contour sang ROI
            points[:, 1] = points[:, 1] - rect[1]
            delta = np.mean(np.abs(laplacian * parking_mask[ind]))
            status = delta < dict['park_laplacian_th']
            # Nếu phát hiện có sự thay đổi thì lưu thời gian lại
            if status != parking_status[ind] and parking_buffer[ind] == None:
                parking_buffer[ind] = video_cur_pos
                change_pos = video_cur_pos

            # Nếu trạng thái vẫn khác với cái đã được lưu và counter đang open
            elif status != parking_status[ind] and parking_buffer[ind] != None:
                if video_cur_pos - parking_buffer[ind] > dict['park_sec_to_wait']:
                    parking_status[ind] = status
                    parking_buffer[ind] = None
            # Nếu trạng thái vẫn như vậy và counter đang open
            elif status == parking_status[ind] and parking_buffer[ind] != None:
                parking_buffer[ind] = None

    # Thay đổi màu và trạng thái hiển thị trên section phía trên
    if dict['parking_overlay']:
        for ind, park in enumerate(parking_data):
            points = np.array(park['points'])
            if parking_status[ind]:
                color = (0, 255, 0) #BGR
                rect = parking_bounding_rects[ind]
                roi_gray_ov = frame_gray[rect[1]:(rect[1] + rect[3]),
                              rect[0]:(rect[0] + rect[2])]  # crop ROI để tính toán nhanh hơn
                res = run_classifier(roi_gray_ov, ind)
                current_count += 1
                if res:
                    parking_data_motion.append(parking_data[ind])
                    color = (0, 0, 255) #BGR
            else:
                color = (0, 0, 255) #BGR

            cv2.drawContours(frame_out, [points], contourIdx=-1,
                             color=color, thickness=2, lineType=cv2.LINE_8)

            if dict['show_ids']:
                print_parkIDs(park, points, frame_out)
        # Hiển thị số lot trống trong frame
        cv2.putText(frame_out, 'Vacant spots in frame: ' + str(current_count), (7, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.728,
                    (98, 189, 184), 2, cv2.LINE_AA)

    if dict['print_out']:
        count = current_count
        file = open("test.txt", "w")
        file.write(str(count))
        file.close()

    if parking_data_motion != []:
        for index, park_coord in enumerate(parking_data_motion):
            points = np.array(park_coord['points'])
            color = (0, 0, 255)
            recta = parking_bounding_rects[ind]
            roi_gray1 = frame_gray[recta[1]:(recta[1] + recta[3]),
                        recta[0]:(recta[0] + recta[2])]  # crop ROI để tính toán nhanh hơn
            fgbg1 = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)
            roi_gray1_blur = cv2.GaussianBlur(roi_gray1.copy(), (5, 5), 3)
            fgmask1 = fgbg1.apply(roi_gray1_blur)
            bw1 = np.uint8(fgmask1 == 255) * 255
            bw1 = cv2.erode(bw1, kernel_erode, iterations=1)
            bw1 = cv2.dilate(bw1, kernel_dilate, iterations=1)
            (_, cnts1, _) = cv2.findContours(bw1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Áp loop cho contours
            for c in cnts1:
                print(cv2.contourArea(c))
                # Nếu contours quá nhỏ thì bỏ qua
                if cv2.contourArea(c) < 4:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                classifier_result1 = run_classifier(roi_gray1, index)
                if classifier_result1:
                    color = (0, 0, 255)  # Đỏ nếu có xe BGR
                else:
                    color = (0, 255, 0)
            classifier_result1 = run_classifier(roi_gray1, index)
            if classifier_result1:
                color = (0, 0, 255)  # Đỏ nếu có xe
            else:
                color = (0, 255, 0)
            cv2.drawContours(frame_out, [points], contourIdx=-1, color=color, thickness=2, lineType=cv2.LINE_8)

    if dict['pedestrian_detection']:
        # Detecr người trong video, sẽ làm giản tốc độ chương trình vì cần một GPU có tốc độ xử lý cao
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
        # Vẽ bounding box
        for (x, y, w, h) in rects:
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # write output frames
    if dict['save_video']:
        out.write(frame_out)

    # Hiển thị video
    cv2.imshow('frame', frame_out)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('c'):
        cv2.imwrite('frame%d.jpg' % video_cur_frame, frame_out)
    elif k == ord('j'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_cur_frame + 1000)  # Nhảy 1000 frames
    elif k == ord('u'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_cur_frame + 500)  # Nhảy 500 frames
    if cv2.waitKey(33) == 27:
        break

cv2.waitKey(0)
cap.release()
if dict['save_video']: out.release()
cv2.destroyAllWindows()