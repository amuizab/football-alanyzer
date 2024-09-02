from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()


    def detect_frames(self, frames):
        detections = []

        batch_size = 20

        for i in range (0,len(frames), batch_size):
            batch = frames[i:i+batch_size]
            results = self.model.predict(batch, conf=0.1, device='cuda')

            detections += results
            #break
        return detections


    def object_tracks(self, frames):
        detections = self.detect_frames(frames)

        self.tracker.reset()

        tracks = {'players':[], 'referees':[], 'ball':[]}

        for frame, detection in enumerate(detections):

            class_names = detection.names
            class_names_inv = {v: k for k, v in class_names.items()}

            detection_byte_track = sv.Detections.from_ultralytics(detection)

            for object_ind, class_id in enumerate(detection_byte_track.class_id):
                if class_names[class_id] == 'goalkeeper':
                    detection_byte_track.class_id[object_ind] = class_names_inv['player']
                    #print('triggerdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd')

            detection_byte_track_tracks = self.tracker.update_with_detections(detection_byte_track)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for i in detection_byte_track_tracks:
               bbox = i[0].tolist()
               cls_id = i[3]
               #cls_name = class_names[cls_id]
               track_id = i[4]

               if cls_id == 2:
                   tracks['players'][frame][track_id] = {'bbox':bbox}
               if cls_id == 3:
                   tracks['referees'][frame][track_id] = {'bbox':bbox}

            for i in detection_byte_track:
               bbox = i[0].tolist()
               cls_id = i[3]

               if cls_id == 0:
                   tracks['ball'][frame][1] = {'bbox':bbox}


            print(detection_byte_track_tracks)

        return tracks
