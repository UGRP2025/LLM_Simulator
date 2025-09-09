import csv
import matplotlib.pyplot as plt
import argparse

def plot_csv_data(file1, file2, label1='Dataset 1', label2='Dataset 2', x_col='positions_X', y_col='positions_y'):
    """
    Reads two CSV files with x and y values and plots them on the same graph.
    
    Args:
        file1 (str): Path to the first CSV file.
        file2 (str): Path to the second CSV file.
        label1 (str): Label for the first dataset.
        label2 (str): Label for the second dataset.
        x_col (str): Name of the column for X values.
        y_col (str): Name of the column for Y values.
    """
    def read_csv(file):
        """Reads x and y data from a CSV file, automatically detecting headers."""
        x, y = [], []
        with open(file, 'r', newline='') as f:
            try:
                # Sniff for header, then reset file pointer
                has_header = csv.Sniffer().has_header(f.read(2048))
                f.seek(0)
            except csv.Error:
                # Could not determine; assume no header for safety
                has_header = False
                f.seek(0)

            if has_header:
                # File has a header, use DictReader
                reader = csv.DictReader(f)
                if x_col not in reader.fieldnames or y_col not in reader.fieldnames:
                    raise ValueError(f"The file {file} has a header but does not contain '{x_col}' or '{y_col}' columns.")
                
                for row in reader:
                    try:
                        x.append(float(row[x_col]))
                        y.append(float(row[y_col]))
                    except (ValueError, KeyError):
                        # Skip rows that have conversion errors or are malformed
                        # print(f"Skipping malformed row in {file}: {row}")
                        continue
            else:
                # No header, use standard reader and assume first two columns
                print(f"No header detected in {file}. Assuming first column is X and second is Y.")
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        try:
                            x.append(float(row[0]))
                            y.append(float(row[1]))
                        except ValueError:
                            # Skip rows with non-numeric data
                            # print(f"Skipping non-numeric row in {file}: {row}")
                            continue
        return x, y
    
    # Read data from both files
    x1, y1 = read_csv(file1)
    x2, y2 = read_csv(file2)
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(x1, y1, label=label1, marker='o', linestyle='-')
    plt.plot(x2, y2, label=label2, marker='x', linestyle='--')
    
    # Add labels, legend, and title
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('Comparison of Car Positions')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compares two CSV files by plotting their X and Y values.")
    parser.add_argument("file1", help="Path to the first CSV file (e.g., reference path).")
    parser.add_argument("file2", help="Path to the second CSV file (e.g., actual path).")
    parser.add_argument("--label1", default="Reference Path", help="Label for the first dataset in the plot.")
    parser.add_argument("--label2", default="Actual Path", help="Label for the second dataset in the plot.")
    parser.add_argument("--x_col", default="positions_X", help="Name of the column for X values.")
    parser.add_argument("--y_col", default="positions_y", help="Name of the column for Y values.")

    args = parser.parse_args()

    plot_csv_data(args.file1, args.file2, 
                  label1=args.label1, label2=args.label2, 
                  x_col=args.x_col, y_col=args.y_col)
