import yaml
import numpy as np

def load_config(config_file):
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {config_file} not found.")
        return None
    return config

class UAV: 
    
    def __init__ (self, cfg_file):

        self.config = load_config(cfg_file)

        # Camera parameter
        self.fx = self.config['camera_focal_length'][0]
        self.fy = self.config['camera_focal_length'][1]
        self.cx = self.config['camera_optical_center'][0]
        self.cy = self.config['camera_optical_center'][1]

        self.intrinsic = np.array([[ self.fx,       0,  self.cx],
                                   [       0, self.fy,  self.cy],
                                   [       0,       0,        1]], dtype = np.float32)
        
        self.diff = np.array(self.config['camera_distortion_coefficients'], dtype = np.float32)

        # Initial state parameter
        self.initial_position = self.config['uav_position'] 
        self.initial_orientation = self.config['uav_orientation']
        self.initial_velocity = self.config['uav_velocity']

    def get_camera_parameter(self):

        return self.intrinsic, self.diff

    def get_init_value(self):
        state = [self.initial_position[0], self.initial_position[1], self.initial_position[2], self.initial_orientation[2]]
        variance = np.identity(4) * 0.1
        return state, variance, self.initial_velocity

    def show_cfg(self):

        for key in self.config.keys():
            
            print(f"{key} {self.config[key]}")

if __name__ == "__main__":

    cfg_path = "uav_config.yml"

    uav = UAV(cfg_path)

    uav.show_cfg()