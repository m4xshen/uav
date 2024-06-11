import csv
import numpy as np

def read_csv(csv_file_path):
    data_list = []
    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  
        for row in csv_reader:
            data_list.append(list(map(float, row[:3])))
    return data_list

def find_clostest(benchmark_data, pred_pose):

    min_idx = 0
    min_error = np.Inf
    min_error_x = np.Inf
    min_error_y = np.Inf
    min_error_z = np.Inf
    pred_x = pred_pose[0]
    pred_y = pred_pose[1]
    pred_z = pred_pose[2]
    for idx, benchmark_pose in enumerate(benchmark_data):
        bm_x = benchmark_pose[0]
        bm_y = benchmark_pose[1]
        bm_z = benchmark_pose[2]

        error = np.sqrt(((bm_x - pred_x) ** 2) +  ((bm_y - pred_y) ** 2)  + ((bm_z - pred_z) ** 2))
        if error < min_error:
            min_idx = idx
            min_error = (error)
            min_error_x = np.abs(bm_x - pred_x)
            min_error_y = np.abs(bm_y - pred_y)
            min_error_z = np.abs(bm_z - pred_z)
    
    return min_idx, (min_error_x, min_error_y, min_error_z)

def calculate_error(benchmark_data, pred_data):

    error_x = 0
    error_y = 0
    error_z = 0
    for pred_pose in pred_data:
        _, (err_x, err_y, err_z) = find_clostest(benchmark_data, pred_pose)
        error_x += err_x
        error_y += err_y
        error_z += err_z

    avg_error_x = error_x / len(pred_data)
    avg_error_y = error_y / len(pred_data)
    avg_error_z = error_z / len(pred_data)
    return avg_error_x, avg_error_y, avg_error_z

if __name__ == "__main__":
    
    BENCHMARK_PATH = "../benchmark.csv"
    PRED_PATH = "./object2.csv"
    
    benchmark_data = read_csv(BENCHMARK_PATH)
    pred_data = read_csv(PRED_PATH)

    avg_err_x, avg_err_y, avg_err_z = calculate_error(benchmark_data, pred_data) 
    print(f"Average error:\nx: {avg_err_x} m\ny: {avg_err_y} m\nz: {avg_err_z} m" )
