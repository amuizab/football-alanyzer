import cv2

def idTracknObjCls(frame, track_id, obj_name, x_min, y_min):
    #Draw id_track and obj class
    text_position = (int(x_min), int(y_min) - 10)
    text_position2 = (int(x_min), int(y_min) + 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 255, 255)  #
    thickness = 2
    cv2.putText(frame, str(track_id), text_position, font, font_scale, color, thickness)
    cv2.putText(frame, str(obj_name), text_position2, font, font_scale, color, thickness)


def drawxyxy(bgr, bbox, frame):
    #Draw bbox
    x_min, y_min, x_max, y_max = bbox
    top_left = (int(x_min), int(y_min))
    bottom_right = (int(x_max), int(y_max))

    cv2.rectangle(frame, top_left, bottom_right, bgr, 2)

    return x_min, y_min, x_max, y_max


def draw_bbox(video, tracks, team_labels):
    #video_draw = []
    for frame_idx in range (0,len(video)): #len of video

        for track_id_ref in tracks['referees'][frame_idx]:
            bbox = tracks['referees'][frame_idx][track_id_ref]['bbox']

            bgr = (255, 255, 255) #black

            x_min, y_min, x_max, y_max = drawxyxy(bgr, bbox, video[frame_idx])
            idTracknObjCls(video[frame_idx], track_id_ref, 'referees', x_min, y_min)

        for track_id_ball in tracks['ball'][frame_idx]:
            bbox = tracks['ball'][frame_idx][track_id_ball]['bbox']

            bgr = (255, 255, 255) #black

            x_min, y_min, x_max, y_max = drawxyxy(bgr, bbox, video[frame_idx])
            idTracknObjCls(video[frame_idx], track_id_ball, 'ball', x_min, y_min)

        for id, track_id in enumerate(tracks['players'][frame_idx]): #get track_id
            #print('track_id',track_id)

            bbox = tracks['players'][frame_idx][track_id]['bbox']

            if team_labels[frame_idx][id] == 0:
                bgr = (0, 0, 255) #red
            else:
                bgr = (255, 0, 0) #blue

            x_min, y_min, x_max, y_max = drawxyxy(bgr, bbox, video[frame_idx])

            idTracknObjCls(video[frame_idx], track_id, 'players', x_min, y_min)

            #print(bbox)


def draw_pitch_kps(video, results_kps):
    #results_kps[frame_id].keypoints.xy.tolist[0]
    video_draw_kps = []
    for frame_idx in range (0,len(video)): #len of video
        for kps in results_kps[frame_idx].keypoints.xy[0].tolist(): #get Keypoint

            x,y = kps
            if x != 0.0 and y != 0.0:
                print(kps)
                cv2.circle(video[frame_idx], (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=2)