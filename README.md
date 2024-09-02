# Football Analyszer

Our system processes input football match videos using advanced computer vision techniques. We employ YOLO (You Only Look Once) for real-time detection of players, referees, and the ball. Subsequently, we utilize ByteTrack to track these detected objects across video frames, ensuring consistent identification throughout the match. The final output is a video that features bounding boxes around the detected objects, clearly distinguishing players, referees, and the ball. Additionally, we generate a dynamic minimap, similar to those seen in video games, providing a top-down view of the match progression. This comprehensive approach combines detection, tracking, and visualization to offer an enhanced football match analysis experience.

## Pipeline
1. Train object detection model:
    - Players
    - Referees
    - Ball
    (Using bounding boxes)
2. Train football field keypoints detection model
3. Video processing:
    - Convert video capture into sequence of frames: video_frame[0: ..]
4. Object tracking:
    - Apply ByteTrack to the trained object detection model
    - Feed video frames into the ByteTrack model
5. Keypoint detection:
    - Use trained model to detect football field keypoints
6. Minimap calibration and object plotting:
    - Calibrate default minimap with detected keypoints
    - Plot detected objects on minimap frame by frame
7. Minimap video creation:
    - Save output minimap frames with plotted objects as a video
8. Original video annotation:
    - Draw bounding boxes for each detected object on every video frame
    - Draw detected keypoints on every video frame
9. Final output:
    - Save annotated original video as result (video frames and minimap frames)
