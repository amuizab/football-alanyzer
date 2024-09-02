from ultralytics import YOLO

import numpy as np
import cv2

from sklearn.cluster import KMeans
from skimage import transform


def keypoints_detect(model_path, video_path):
    model = YOLO(model_path)
    return model.predict(source=video_path)




def xy2smallbbox(x, y, width, height):
    xmin = x - width / 2
    ymin = y - height / 2
    xmax = x + width / 2
    ymax = y + height / 2
    return xmin, ymin, xmax, ymax

def my_average_color(image):
    #np_image = np.array(image)
    np_image = image
    average_color = np.mean(np_image, axis=(0, 1))
    return average_color

def mini_pitch_id(kps_id):

    # my_dict = {str(i): [] for i in range(32)} -> this can be useful if have different mini pitch and input from user instad hard coded
    # print(my_dict)
    dst = []
    pitch_points = {'0':[34,50], '1':[34,135], '2':[34,192], '3':[34,287], '4':[34,345], '5':[34,430],
                    '6':[62,192], '7':[62,287], '8':[90,240], '9':[120,135], '10':[120,200], '11':[120,275],
                    '12':[120,345], '13':[320,50], '14':[320,192], '15':[320,287], '16':[320,430], '17':[520,135],
                    '18':[520,200], '19':[520,275], '20':[520,345], '21':[548,240], '22':[578,192], '23':[578,287],
                    '24':[606,50], '25':[606,135], '26':[606,192], '27':[606,287], '28':[606,345], '29':[606,430],
                    '30':[272,239], '31':[367,239]}

    for index in kps_id:
        dst.append(pitch_points[str(int(index))])

    return dst

def get_perspective_matrix(src, dst):
    #print('disinii bung')
    return transform.estimate_transform('projective', np.float32(src), np.float32(dst))

def transform_point(obj_points, matrix): #this is point from bbox (object in the field)

    new_points = []
    #print(obj_points[0])
    for xyxy in obj_points:
        x_min, y_min, x_max, y_max = xyxy[0]
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2


        #point_reshaped = np.array([x_center, y_center, 1]).reshape(3, 1)
        point_reshaped = np.array([x_center, y_center, 1])
        new_point = np.dot(matrix, point_reshaped)
        new_point = new_point / new_point[2]

        new_points.append([int(new_point[0]), int(new_point[1])])

    return new_points

def points_draw(image, new_points, color1, color2, labels_team):

    new_img = image.copy()
    #print(new_points)

    for id, point in enumerate(new_points):

        if labels_team[id] == 0:
            bgr = color1
        else:
            bgr = color2

        cv2.circle(new_img, (point[0], point[1]), radius=5, color=bgr, thickness=2)

    return new_img

def ball_draw(image, points):

    if len(points) == 0:
        pass

    else:
        bgr = (0, 0, 0) #black
        cv2.circle(image, (points[0][0], points[0][1]), radius=2, color=bgr, thickness=2)

def calibrate_pitch(image, pitch_kps, tracks, video):

    ##
    # pitch_kps and tracks need to be in the correct format
    #code here
    ##
    labels = []
    pitch_frames = []
    index = np.arange(len(pitch_kps[0].keypoints.xy[0])).reshape(len(pitch_kps[0].keypoints.xy[0]), 1)
    image = cv2.imread(image)
    #print('triggreeeeed')


    for frame_idx in range(0, len(pitch_kps)):
        #print('triggreeeeed')
        src = []
        kps_id = []
        bboxes = []
        bboxesball = []
        #print('triggreeeeed')
        id_keypoints = np.hstack((pitch_kps[frame_idx].keypoints.xy.tolist()[0], index))
        #print('triggreeeeedusss')
        for i in id_keypoints:
            if i[0] != 0 and i[1] != 0:     # eliminate keypoints that not detected (0 xy values)
                src.append([i[0], i[1]])
                kps_id.append(i[2])

        dst = mini_pitch_id(kps_id)         #get the mini map destination points
        #print(src)
        #print(dst)
        matrix = get_perspective_matrix(src, dst)
        #print('bersahilll')
        for track_id in tracks['players'][frame_idx]:

            bboxes.append([tracks['players'][frame_idx][track_id]['bbox']])

        for track_id_ball in tracks['ball'][frame_idx]:
            bboxesball.append([tracks['ball'][frame_idx][track_id_ball]['bbox']])

        #print('TA')


        new_bbox = [] #bbox for new rect
        avg_colors = []
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox[0]
            x = (xmin + xmax) // 2
            y = (ymin + ymax) // 2

            xmin, ymin, xmax, ymax = xy2smallbbox(x, y, 5, 5)
            new_bbox.append([int(xmin), int(ymin), int(xmax), int(ymax)])
            cropped_image = video[frame_idx][int(ymin):int(ymax), int(xmin):int(xmax)]
            avg_colors.append(my_average_color(cropped_image))

        if frame_idx == 0:
            kmeans = KMeans(n_clusters=2, random_state=0)
            kmeans.fit(avg_colors)
            label = kmeans.labels_
        else:
            label = kmeans.predict(avg_colors)


        if frame_idx == 0:
            for id, labl in enumerate(label):
                #print('hello')
                if labl == 0:
                    c0 = avg_colors[id]
                if labl == 1:
                    c1 = avg_colors[id]


        new_points = transform_point(bboxes, matrix) # dimasukin boxesball dibelakang, setelah dapet noew points, diPOP atau di sliceing aja  newpoitns[-1]utk ball, nwpoints[0:x] utk players
        new_points_ball = transform_point(bboxesball, matrix)
        #break
        pitch_frame = points_draw(image, new_points, c0, c1, label)
        ball_draw(pitch_frame, new_points_ball)
        pitch_frames.append(pitch_frame)
        labels.append(label)

    return pitch_frames, labels