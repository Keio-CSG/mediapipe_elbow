import pyrealsense2 as rs
import numpy as np
import mediapipe as mp
import cv2

def main():
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    try:
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            while True:
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not depth_frame or not color_frame:
                    print("********* frame is dropped **********")
                    continue

                image = np.asanyarray(color_frame.get_data())
                image = cv2.flip(image, 1)
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_image = cv2.flip(depth_image, 1)

                results = pose.process(image)

                if results.pose_landmarks:
                    landmark_points = []
                    image_width, image_height = image.shape[1], image.shape[0]
                    for index, (landmark, world_landmark) in enumerate(zip(
                        results.pose_landmarks.landmark, 
                        results.pose_world_landmarks.landmark)):
                        if landmark.x < 0 or landmark.x > 1 or landmark.y < 0 or landmark.y > 1:
                            landmark_points.append(None)
                            continue
                        landmark_x_px = int(landmark.x * image_width)
                        landmark_y_px = int(landmark.y * image_height)
                        landmark_points.append((
                            (landmark_x_px, landmark_y_px),
                            [world_landmark.x, world_landmark.y, depth_image[landmark_y_px][landmark_x_px] / 1000]
                        ))
                        if index in [12,14,16]:
                            cv2.circle(image, (landmark_x_px, landmark_y_px), 5, (0, 255, 0), 2)

                    if landmark_points[12] is not None and landmark_points[14] is not None and landmark_points[16] is not None:
                        cv2.line(image, landmark_points[12][0], landmark_points[14][0], (0, 255, 0), 2)
                        cv2.line(image, landmark_points[14][0], landmark_points[16][0], (0, 255, 0), 2)
                        p_shoulder = np.array(landmark_points[12][1])
                        p_elbow = np.array(landmark_points[14][1])
                        p_wrist = np.array(landmark_points[16][1])
                        upperarm = p_shoulder - p_elbow
                        forearm = p_wrist - p_elbow
                        angle = np.arccos(np.dot(upperarm, forearm) / ( np.linalg.norm(upperarm) * np.linalg.norm(forearm) ))
                        angle_deg = angle * 180 / np.pi
                        image = cv2.putText(image, f"{int(angle_deg)} degree",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0),
                            3, cv2.LINE_AA)
                cv2.imshow('MediaPipe Pose', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()