from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        

    def detect_frames(self, frames):
        detections = []
        detections2 = []
        detections3 = []
        detections4 = []
        batch_size = 20

        for i in range (0,len(frames), batch_size):
            batch = frames[i:i+batch_size]
            results = self.model.predict(batch, conf=0.1, device='cpu')
            
            detections += results
            detections2 += results[0].boxes.data.tolist()
            detections3.extend(results)
            detections4.extend(results[0].boxes.data.tolist())
            break
        return detections, detections2, detections3, detections4


    def object_tracks(self, frames):
        detections, detections2, detections3, detections4 = self.detect_frames(frames)

        for frame, detection in enumerate(detections):

            class_names = detection.names
            class_names_inv = {v: k for k, v in class_names.items()}

            detection_byte_track = sv.Detections.from_ultralytics(detection)

            for object_ind, class_id in enumerate(detection_byte_track.class_id):
                if class_names[class_id] == 'goalkeeper':
                    detection_byte_track.class_id[object_ind] = class_names_inv['player']
            
            detection_byte_track_tracks = self.tracker.update_with_detections(detection_byte_track)

            print(detection_byte_track_tracks)


            

    