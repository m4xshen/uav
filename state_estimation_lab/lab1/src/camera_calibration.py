import os
import cv2
import numpy as np

# Define chess board pattern size
pattern_size = (8, 6)

# Criteria for termination of the iterative process of corner detection
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Load images
images_folder = "./image"
images_list = os.listdir(images_folder)

for fname in images_list:

    img_path = os.path.join(images_folder, fname)
    gray = cv2.imread(img_path, 0)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        gray = cv2.drawChessboardCorners(gray, pattern_size, corners2, ret)

        # Resize the window to half the size of the image
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', (gray.shape[1] // 2, gray.shape[0] // 2))
        cv2.imshow('img', gray)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the camera matrix
print(f"Objective function value: {ret}\n")
print(f"Distorition coefficient: {dist}\n")
print("Camera Matrix:")
print(mtx)

fx = mtx[0, 0] 
fy = mtx[1, 1] 
cx = mtx[0, 2] 
cy = mtx[1, 2] 

print(f"\nfx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
