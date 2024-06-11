from djitellopy import Tello
from pupil_apriltags import Detector
from task import Task1, Task2
import cv2


def main():
    drone = Tello()
    drone.connect()
    print(drone.get_battery())
    drone.streamon()
    # drone.takeoff()

    detector = Detector(families="tag36h11")

    task1 = Task1(drone)
    task1_finished = False
    task2 = Task2(drone)
    task2_finished = False

    while True:
        frame = drone.get_frame_read().frame
        tag_list = detector.detect(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            estimate_tag_pose=True,
            camera_params=[904, 905, 482, 350],
            tag_size=0.1,
        )

        if not task1_finished:
            task1_finished = task1.run(tag_list, frame)

            if task1_finished:
                drone.send_rc_control(0, 0, 0, 0)
        elif not task2_finished:
            task2_finished = task2.run(tag_list, frame)

            if task2_finished:
                drone.send_rc_control(0, 0, 0, 0)

        for tag in tag_list:
            for corner in tag.corners:
                cv2.circle(frame, list(map(int, corner)), 5, (0, 0, 255), -1)

        cv2.imshow("drone", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(30)
        if (key & 0xFF) == ord("q"):
            break

    cv2.destroyAllWindows()
    # drone.land()
    drone.streamoff()


if __name__ == "__main__":
    main()
