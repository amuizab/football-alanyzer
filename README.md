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
  

## Demo
- Input Video
  ![testvideo_compressd_gif](https://github.com/user-attachments/assets/b0c20658-19bf-4184-b74b-f7d26ff39e4d)

- Output Video
  ![output-lean-gif](https://github.com/user-attachments/assets/d3bffc62-021a-42af-983b-fe46b47881ef)

- Minimap Output
  ![outputmini-map-gif](https://github.com/user-attachments/assets/f98df5a2-7712-4f4d-b2ac-df0872bc61ae)


## Next Development
    - Action detection to detect what is happening on the pitch and record it as the stats of the game.
    - Provide statistical analysis (pass completion, shot accuracy, player distance, and so)
    - Tactical Analysis on formation changing throughout the match
  
