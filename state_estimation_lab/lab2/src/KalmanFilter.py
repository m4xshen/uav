import numpy as np

class KalmanFilter:

  def __init__(self, x0, s0):

    # Matrix A describe how the state evolves from t-1 to t without control or noise
        self.A = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        # Matrix B describes how the control input changes the state from t-1 to t
        self.B = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Matrix C describes how to map the state xt to an observation zt
        self.C = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        # Matrix R describes the motion noise covariance
        self.R = np.eye(4)
        # Matrix Q describes the observation noise covariance
        self.Q = np.eye(4)
        # State estimate
        self.S = s0 # Initial state covariance
        self.x = x0 # Initial state

  def predict(self, u = None):
      if u is None:
            u = np.zeros((4,))
      # Predict the next state (prior)
      self.x = np.dot(self.A, self.x) + np.dot(self.B, u)

      # Predict the next covariance (prior)
      self.S = np.dot(np.dot(self.A, self.S), self.A.T) + self.R

  def update(self, z):

      # Compute Kalman gain 
        S_hat = np.dot(np.dot(self.C, self.S), self.C.T) + self.Q
        K = np.dot(np.dot(self.S, self.C.T), np.linalg.inv(S_hat))

        # Measurement prediction
        observe_pred = self.C @ self.x
        y = z - observe_pred        

        angle_diff = y[3]
        if z[3] < 0 and observe_pred[3] > 0:
          y[3] = -((angle_diff + np.pi) % (2 * np.pi) - np.pi)

        elif z[3] > 0 and observe_pred[3] < 0:
          y[3] = ((angle_diff + np.pi) % (2 * np.pi) - np.pi)

        # Update state estimate
        self.x = self.x + np.dot(K, y)

        # Update estimate error covariance
        self.S = self.S - np.dot(np.dot(K, self.C), self.S)

  def get_current_state(self):

      return self.x, self.S





