import rclpy
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
import numpy as np 
import matplotlib.pyplot as plt 
import networkx as nx
import yaml
from PIL import Image 
import csv



def load_yaml(yaml_path):
    
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)
    
def pgm_to_binary_occupancy(pgm_path):

    img = Image.open(pgm_path)  

    grid = np.array(img)
    occupancy_grid = grid 

    for y in range(occupancy_grid.shape[0]):
        for x in (range(occupancy_grid.shape[1])):

            if grid[y][x]== 254:
                occupancy_grid[y][x] =  0
            else:
                occupancy_grid[y][x] = 1
    
    return occupancy_grid



def create_occupancy_grid(occupany_grid, yaml_data):
    race_map = MapMetaData()
    race_map.resolution = yaml_data['resolution']
    race_map.width = occupany_grid.shape[1]
    race_map.height = occupany_grid.shape[0]
    race_map.origin = Pose()
    race_map.origin.position.x = yaml_data['origin'][0]
    race_map.origin.position.y = yaml_data['origin'][1]
    race_map.origin.position.z = 0.0
    race_map.origin.orientation.w = float(yaml_data['origin'][2])
    occupied_thresh = yaml_data['occupied_thresh']
    free_thresh = yaml_data['free_thresh']


    for y in range(race_map.height):

        for x in range(race_map.width):

            if occupany_grid[y][x] >= occupied_thresh:
                occupany_grid[y][x] = 0

            elif occupany_grid[y][x] <= free_thresh:
                occupany_grid[y][x] = 1



    return  occupany_grid


def create_CSV(CSV):
        with open("occupancy_grid.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for row in CSV:
                 writer.writerow(row)


def Visulaize_CSV():

    with open("occupancy_grid.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        occupancy_grid = np.array([list(map(float, row)) for row in reader])

# Visualize the occupancy grid
    plt.imshow(occupancy_grid, cmap="viridis", origin="upper")
    plt.colorbar(label="Occupancy Probability")
    plt.title("Occupancy Grid Visualization")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()



def main():

    yaml_data = load_yaml('/home/autodrive_devkit/src/car_control/car_control/maps/iros_2024/scan_iros_map_compete2024.yaml')
    occupancy_grid = pgm_to_binary_occupancy('/home/autodrive_devkit/src/car_control/car_control/maps/iros_2024/scan_iros_map_compete2024.pgm')


    #occupancy_grid = create_occupancy_grid(occupancy_grid,yaml_data)
    
    create_CSV(occupancy_grid)
    Visulaize_CSV()




if __name__ ==  '__main__':

    main()
