import csv
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from pupil_apriltags import Detector
from kalman import KalmanFilter

def centroid_batch(bboxes1: np.ndarray, bboxes2: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Compute the normalized Euclidean distances between the centroids of two batches of bounding boxes.

    Parameters:
    - bboxes1: np.ndarray of shape (N, 4), where each row represents a bounding box [x1, y1, x2, y2].
    - bboxes2: np.ndarray of shape (M, 4), where each row represents a bounding box [x1, y1, x2, y2].
    - w: int, the width of the image (used for normalization).
    - h: int, the height of the image (used for normalization).

    Returns:
    - normalized_distances: np.ndarray of shape (N, M), where the element at (i, j) is the normalized Euclidean distance
      between the centroid of the i-th bounding box in bboxes1 and the centroid of the j-th bounding box in bboxes2.
    """
    # TODO
    # Compute centroids of bboxes1
    centroids1 = np.array([
        (bboxes1[:, 0] + bboxes1[:, 2]) / 2,
        (bboxes1[:, 1] + bboxes1[:, 3]) / 2
    ]).T

    # Compute centroids of bboxes2
    centroids2 = np.array([
        (bboxes2[:, 0] + bboxes2[:, 2]) / 2,
        (bboxes2[:, 1] + bboxes2[:, 3]) / 2
    ]).T

    # Initialize the distance matrix
    distances = np.zeros((bboxes1.shape[0], bboxes2.shape[0]))

    # Compute Euclidean distances between each pair of centroids
    for i in range(bboxes1.shape[0]):
        for j in range(bboxes2.shape[0]):
            dist = np.sqrt((centroids1[i, 0] - centroids2[j, 0]) ** 2 + 
                           (centroids1[i, 1] - centroids2[j, 1]) ** 2)
            distances[i, j] = dist

    # Normalize the distances by the dimensions of the image
    normalized_distances = distances / np.sqrt(w**2 + h**2)
    
    return normalized_distances

class Tracker:
    shared_kalman = KalmanFilter()

    def __init__(self, max_age=30, min_hits=3, max_iou_distance=0.7):
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_iou_distance = max_iou_distance
        self.track_id = 0
        self.tracks = []

    def update(self, detections, frame_size):
        # Predict the current state with the Kalman Filter
        if len(self.tracks) > 0:
            for track in self.tracks:
                track["mean"], track["covariance"] = self.shared_kalman.predict(track["mean"], track["covariance"])

        matched, unmatched_detections, unmatched_tracks = self.match_tracks(detections, frame_size)

        # Update matched tracks with new detections
        for track_idx, detection_idx in matched:
            track = self.tracks[track_idx]
            detection = detections[detection_idx]
            mean, covariance = self.shared_kalman.update(track["mean"], track["covariance"], detection[:4])
            track["mean"], track["covariance"] = mean, covariance
            track["hits"] += 1
            track["age"] = 0
            track["state"] = "confirmed" if track["hits"] >= self.min_hits else "tentative"
            track["det"] = detection

        # Create new tracks for unmatched detections
        for idx in unmatched_detections:
            detection = detections[idx]
            mean, covariance = self.shared_kalman.initiate(detection[:4])
            self.tracks.append({
                "mean": mean,
                "covariance": covariance,
                "track_id": self.track_id,
                "hits": 1,
                "age": 0,
                "state": "tentative",
                "det": detection
            })
            self.track_id += 1

        # Mark unmatched tracks as 'deleted' if they exceed max_age
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            track["age"] += 1
            if track["age"] > self.max_age:
                track["state"] = "deleted"

        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if t["state"] != "deleted"]

        confirmed_tracks = [np.append(t["det"], [t["track_id"]]) for t in self.tracks if t["state"] == "confirmed"]

        return np.array(confirmed_tracks)

    def match_tracks(self, detections, frame_size):
        if len(self.tracks) > 0 and len(detections) > 0:
            track_boxes = np.array([track["mean"][:4] for track in self.tracks])
            detection_boxes = detections[:, :4]
            distances = centroid_batch(track_boxes, detection_boxes, frame_size[0], frame_size[1])
            matched_indices, unmatched_detections, unmatched_tracks = self.assign_detections_to_tracks(distances)
            return matched_indices, unmatched_detections, unmatched_tracks
        else:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

    def assign_detections_to_tracks(self, distances):
        # Use the Hungarian algorithm (linear_sum_assignment) to find the optimal assignment
        # that minimizes the total distance between tracks and detections.
        row_ind, col_ind = linear_sum_assignment(distances)
        matched_indices = []
        unmatched_detections = set(range(distances.shape[1]))
        unmatched_tracks = set(range(distances.shape[0]))

        # TODO
        # Iterates through the matched pairs of track and detection indices (consider self.max_iou_distance).
        # Adding to matched_indices, and removing from unmatched_detections and unmatched_tracks.
        for r, c in zip(row_ind, col_ind):
            if distances[r, c] < self.max_iou_distance:
                matched_indices.append((r, c))
                unmatched_detections.discard(c)
                unmatched_tracks.discard(r)

        return matched_indices, list(unmatched_detections), list(unmatched_tracks)

def main(video_path):
    detector = Detector(families="tag36h11")
    tracker = Tracker()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Prepare CSV file to store tracking info
    with open('tracking_info_3.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame_id", "apriltag_id", "track_id", "corner1_x", "corner1_y", "corner2_x", "corner2_y", "corner3_x", "corner3_y", "corner4_x", "corner4_y"])

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video file.")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            tag_list = detector.detect(gray_frame)

            detections = []
            for tag in tag_list:
                x_min, y_min = np.min(tag.corners, axis=0)
                x_max, y_max = np.max(tag.corners, axis=0)
                detections.append([x_min, y_min, x_max, y_max, tag.tag_id, tag.corners[0][0], tag.corners[0][1], tag.corners[1][0], tag.corners[1][1], tag.corners[2][0], tag.corners[2][1], tag.corners[3][0], tag.corners[3][1]])

            detections = np.array(detections)

            confirmed_tracks = tracker.update(detections, frame.shape[:2])

            for track in confirmed_tracks:
                x_min, y_min, x_max, y_max, tag_id, corner1_x, corner1_y, corner2_x, corner2_y, corner3_x, corner3_y, corner4_x, corner4_y, track_id = map(int, track)

                cv2.putText(frame, f"ID: {track_id}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.circle(frame, (corner1_x, corner1_y), 5, (0, 0, 255), -1)
                cv2.circle(frame, (corner2_x, corner2_y), 5, (0, 0, 255), -1)
                cv2.circle(frame, (corner3_x, corner3_y), 5, (0, 0, 255), -1)
                cv2.circle(frame, (corner4_x, corner4_y), 5, (0, 0, 255), -1)

                writer.writerow([frame_id, tag_id, track_id, corner1_x, corner1_y, corner2_x, corner2_y, corner3_x, corner3_y, corner4_x, corner4_y])

            cv2.imshow("video", frame)

            key = cv2.waitKey(30)
            if (key & 0xFF) == ord("q"):
                break

            frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # TODO: Replace with the path to your video file
    video_path = "./track_3.MOV" 
    main(video_path)
