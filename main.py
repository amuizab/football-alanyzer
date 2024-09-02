

from utils import read_video, save_video
from tracker import Tracker
from keypoints import keypoints_detect, calibrate_pitch
from draw import draw_bbox, draw_pitch_kps



def main():
    video_path = 'testvideo.mp4'
    pitch_template = 'pitch.jpg'

    #Read Video
    video = read_video(video_path)

    #Apply Tracker
    tracker = Tracker('models/bestylv5.pt')
    tracks = tracker.object_tracks(video)


    #Detect Field Keypoints
    results_kps = keypoints_detect('models/lastposeylv8.pt', video_path)

    #Calibrate Minimap_template and Object Plotting -> extract team labels to differentiate the color, and minimap frames
    mini_map, team_labels = calibrate_pitch(pitch_template, results_kps, tracks, video)


    #Draw Object bbox and Field Keypoints
    draw_bbox(video, tracks, team_labels)
    draw_pitch_kps(video, results_kps)

    
    #Save frames into Videos
    save_video(mini_map, 'outputmini_map_lean.avi')
    save_video(video, 'output_lean.avi')



if __name__ == '__main__':
    main()

