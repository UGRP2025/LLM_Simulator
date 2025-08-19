import csv
import matplotlib.pyplot as plt

def plot_csv_data(file1, file2):
    """
    Reads two CSV files with x and y values and plots them on the same graph without using pandas.
    
    Args:
        file1 (str): Path to the first CSV file.
        file2 (str): Path to the second CSV file.
    """
    def read_csv(file):
        """Car performance"""
        x, y = [], []
        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            if 'positions_X' not in reader.fieldnames or 'positions_y' not in reader.fieldnames:
                raise ValueError(f"The file {file} must contain 'x' and 'y' columns.")
            for row in reader:
                x.append(float(row['positions_X']))
                y.append(float(row['positions_y']))
        return x, y
    
    # Read data from both files
    x1, y1 = read_csv(file1)
    x2, y2 = read_csv(file2)
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(x1, y1, label='Dataset 1', marker='o')
    plt.plot(x2, y2, label='Dataset 2', marker='x')
    
    # Add labels, legend, and title
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Comaprison of car refrence positions and actual positions')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == '__main__':

    plot_csv_data('/home/autodrive_devkit/src/car_control/car_control/CSVs/actual.csv',
                                 '/home/autodrive_devkit/src/car_control/car_control/CSVs/Centerline_points.csv')
