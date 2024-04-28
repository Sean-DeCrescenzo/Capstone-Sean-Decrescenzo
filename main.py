# Student Name: Sean Decrescenzo
# Student ID: 000973102
# Import necessary modules and classes
import csv
import datetime
import math
import os
import sys
import random
import truck
from package import Package
from hashTable import HashTable
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import RadiusNeighborsTransformer

logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

print(f'Student Name: Sean Decrescenzo\n'
      f'Student ID:   000973102')

# Determine the base path based on the script location
if getattr(sys, 'frozen', False):
    # If running from a bundle (executable), use the directory of the executable
    base_path = os.path.dirname(sys.executable)
else:
    # If running from the script, use the directory of the script
    base_path = os.path.dirname(os.path.abspath(__file__))

# Check if the CSV folder exists in the base path
csv_folder_path = os.path.join(base_path, 'CSV')
if not os.path.exists(csv_folder_path):
    # If the CSV folder is not found, navigate up one level
    base_path = os.path.dirname(base_path)
    csv_folder_path = os.path.join(base_path, 'CSV')

# Construct the file paths
csv_path_distance = os.path.join(csv_folder_path, 'WGUPSDistanceFile2.0.csv')
csv_path_address = os.path.join(csv_folder_path, 'WGUPSAddressFile2.0.csv')

# For CSV path override during debugging
csv_path_package = os.path.join(csv_folder_path, 'WGUPSPackageFile2.0.csv')

# Read the csv files of distance and address information
with open(csv_path_distance, 'r') as csvDistanceTable:
    raw_distance_data = list(csv.reader(csvDistanceTable))

with open(csv_path_address, 'r') as csvAddressTable:
    raw_address_data = list(csv.reader(csvAddressTable))

# Initialize the hash table for storing package data with custom hashTable class
package_table = HashTable(10)


# Function to load package data into the hash table
def load_package_data(file_path, package_data_table):
    with open(file_path, 'r') as csvPackageFile:
        csv_package_data = csv.reader(csvPackageFile)
        next(csv_package_data)  # Skip header
        for row in csv_package_data:
            if len(row) >= 6:  # Ensure the row has at least 6 columns
                package_id = row[0].strip()
                address = row[1].strip()
                city = row[2].strip()
                state = row[3].strip()
                zip_code = row[4].strip()
                weight = int(row[5].strip())

                # Check if any of the required columns are empty
                if not (package_id and address and city and state and zip_code and weight):
                    print(f"Ignoring invalid row: {row}")
                    continue

                package = Package(package_id, address, city, state, zip_code, weight, "At Hub", None, None, None)
                package_data_table.insert(int(package.package_id), package)
            else:
                print(f"Ignoring invalid row: {row}")


def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        print(f"Selected package file: {file_path}")
        load_package_data(file_path, package_table)
        root.destroy()


root = tk.Tk()
root.title("File Selection Example")
button = tk.Button(root, text="Select File", command=select_file)
button.pack(padx=40, pady=40)
root.mainloop()

trucks = []
for i in range(0, 4):
    new_truck = truck.Truck(i, i, [], 0.0, "4001 South 700 East", 30, datetime.timedelta(hours=7), 0)
    trucks.append(new_truck)

delivery_order = {}
route_durations = {}


def calculate_distance(address1, address2, distance_data):
    index1 = get_destination_index(address1)
    index2 = get_destination_index(address2)
    if index1 is not None and index2 is not None:
        distance_between_indexes = distance_data[index1][index2]
        if distance_between_indexes == '':
            distance_between_indexes = distance_data[index2][index1]
        return float(distance_between_indexes)
    return float('inf')


def get_destination_index(street_address):
    for i, record in enumerate(raw_address_data):
        if record[2] == street_address:
            return i
    return None


# Gather a list of all street addresses to create distance matrix
address_list = [record[2] for record in raw_address_data]

# Calculate the distances between addresses
distances = np.zeros((len(address_list), len(address_list)))
for i in range(len(address_list)):
    for j in range(i + 1, len(address_list)):
        distance = calculate_distance(address_list[i], address_list[j], raw_distance_data)
        distances[i, j] = distance
        distances[j, i] = distance

# Create a RadiusNeighborsTransformer object and map neighbors as indexes within 10.0 distance
rnt = RadiusNeighborsTransformer(radius=10, mode='distance', metric='precomputed')

# Fit the transformer to the distances
rnt.fit(distances)

# Initialize an empty connectivity matrix
connectivity_matrix = np.zeros((len(address_list), len(address_list)))


# Get the nearest neighbor for a given index and add the data to the connectivity_matrix array
for i in range(len(address_list)):
    neighbors_indices = rnt.radius_neighbors([distances[i]], return_distance=False)[0]
    nearest_neighbors = [address_list[i] for i in neighbors_indices]
    connectivity_matrix[i, neighbors_indices] = 1

# Cluster the addresses
clusterer = AgglomerativeClustering(n_clusters=4, connectivity=connectivity_matrix, linkage='ward')
clusters = clusterer.fit_predict(distances)


# Assign packages to clusters
package_clusters = {}
for package_id, package in package_table.items():
    cluster_id = clusters[get_destination_index(package.address)]
    if cluster_id not in package_clusters:
        package_clusters[cluster_id] = []
    package_clusters[cluster_id].append((package_id, package))


for truck in trucks:
    cluster_id = truck.truck_id
    if cluster_id in package_clusters:
        cluster = package_clusters[cluster_id]
        while cluster and truck.load_weight < truck.max_weight:
            package_id, package = cluster[0]  # Get the first package in the cluster
            if package.weight <= truck.max_weight - truck.load_weight:
                truck.packages.append(str(package_id))  # Append only the package ID
                truck.load_weight += package.weight
                cluster.pop(0)  # Remove the package from the cluster
            else:
                break  # Stop loading packages if weight limit is reached
        if not cluster:  # Remove the cluster from the copy if it's empty
            del package_clusters[cluster_id]

# Fill remaining space in trucks from non-empty clusters
non_empty_clusters = {c_id: packages for c_id, packages in package_clusters.items() if packages}
for truck in trucks:
    if truck.load_weight < truck.max_weight:
        for cluster_id, cluster in package_clusters.items():
            while cluster and truck.load_weight < truck.max_weight:
                package_id, package = cluster[0]  # Get the first package in the cluster
                if package.weight <= truck.max_weight - truck.load_weight:
                    truck.packages.append(str(package_id))  # Append only the package ID
                    truck.load_weight += package.weight
                    cluster.pop(0)  # Remove the package from the cluster
                else:
                    break  # Stop loading packages if weight limit is reached
            if not cluster:  # Remove the cluster from the copy if it's empty
                del package_clusters[cluster_id]


def calculate_route_time(truck, delivery_order, distance_data):
    current_location = "4001 South 700 East"
    total_time = datetime.timedelta()
    for package in delivery_order:
        current_package = package_table.get_package(int(package))
        distance = calculate_distance(current_location, current_package.address, distance_data)
        delivery_time = datetime.timedelta(hours=distance / truck.speed)
        total_time += delivery_time
        current_location = current_package.address
    return total_time


def simulated_annealing_truck_route(truck, distance_data):
    current_order = truck.packages
    original_time = calculate_route_time(truck, current_order, distance_data)
    current_time = calculate_route_time(truck, current_order, distance_data)
    best_order = current_order.copy()
    best_time = current_time
    temperature = 221.25
    cooling_rate = 0.002
    min_temperature = 0.01
    iteration_count = 0
    while temperature > min_temperature:
        new_order = current_order.copy()
        i, j = random.sample(range(len(new_order)), 2)
        new_order[i], new_order[j] = new_order[j], new_order[i]
        new_time = calculate_route_time(truck, new_order, distance_data)
        cost_diff = new_time - current_time
        if cost_diff.total_seconds() < 0 or random.random() < math.exp(-cost_diff.total_seconds() / temperature):
            current_order = new_order.copy()
            current_time = new_time
            if current_time < best_time:
                best_order = current_order.copy()
                best_time = current_time
        temperature *= 1 - cooling_rate
        iteration_count += 1
        route_durations[truck.truck_id].append(best_time)  # Save the total route duration for this iteration

    print(f"New Route Time: {best_time} \n"
          f"OriginalTime: {original_time} \n"
          f"Iterations for Truck {truck.truck_id}: {iteration_count}\n"
          f"packages on Truck {len(truck.packages)}\n")
    return best_order


# Optimize routes using simulated annealing
for truck in trucks:
    route_durations[truck.truck_id] = []
    truck.packages = simulated_annealing_truck_route(truck, raw_distance_data)


def deliver_package(truck, next_package, distance):
    delivery_time = datetime.timedelta(hours=distance / truck.speed)
    truck.accumulated_time += delivery_time
    next_package.loaded_time = truck.departure_time
    next_package.truck_info = truck.truck_id
    next_package.status = "Delivered"
    next_package.delivery_time = truck.departure_time + truck.accumulated_time
    truck.mileage += distance
    truck.location = next_package.address


# Function that delivers all packages on a given truck. This function has
# an overall time complexity of O(N^2*M) due to its utilization of nested loops iterating on data of varying sizes.
# This function has an overall space complexity of O(N+M+P) as it relies on the storage of data of varying sizes and
# types throughout its execution
def deliver_packages(current_truck, distance_data):
    while truck.packages:
        # While there are undelivered packages find the next package to deliver and its distance
        next_package = package_table.get_package(int(truck.packages[0]))
        if next_package is not None:
            distance = calculate_distance(truck.location, next_package.address, distance_data)
            # Deliver the package
            deliver_package(current_truck, next_package, distance)
            current_truck.packages.remove(next_package.package_id)
            delivery_order[truck.truck_id].append(next_package.package_id)
        else:
            # No undelivered packages left for the truck
            break

    # After delivering all packages, return the truck to the hub
    if current_truck.location != "4001 South 700 East":
        # Get the index of the hub and current location
        hub_address = "4001 South 700 East"
        current_location_index = get_destination_index(current_truck.location)
        if current_location_index is not None:
            # Calculate the distance and time to return to the hub
            distance_to_hub = calculate_distance(truck.location, hub_address, distance_data)
            current_truck.mileage += distance_to_hub
            current_truck.accumulated_time += datetime.timedelta(hours=distance_to_hub / current_truck.speed)
            current_truck.location = "4001 South 700 East"
            # Track the time of arrival at the hub
            arrival_time_at_hub = current_truck.departure_time + current_truck.accumulated_time
            arrival_datetime_at_hub = datetime.datetime.now().replace(hour=0, minute=0, second=0,
                                                                      microsecond=0) + arrival_time_at_hub

            print(
                f"Truck {current_truck.truck_id} - Arrival Time at Hub: {arrival_datetime_at_hub.strftime('%I:%M %p')}")


# Deliver packages
for truck in trucks:
    delivery_order[truck.truck_id] = []
    deliver_packages(truck, raw_distance_data)

# Create a matrix to represent the delivery routes
routes_matrix = np.zeros((len(trucks), len(raw_address_data)))
# Fill the matrix with the delivery order for each truck
for truck_id, delivery_order in delivery_order.items():
    for package_id in delivery_order:
        address_index = get_destination_index(package_table.get_package(int(package_id)).address)
        routes_matrix[truck_id - 1][address_index] += 1

# Assuming package_table is a dictionary where keys are addresses and values are delivery counts
# Example: package_table = {'Address 1': 3, 'Address 2': 5, 'Address 3': 2, ...}
frequency_matrix = np.zeros((len(raw_address_data), 1))  # Initialize a matrix with one column for frequencies
for i, address_data in enumerate(raw_address_data):
    address = address_data[2]  # Assuming the address is at index 2
    for package in package_table.get_all_packages():
        if package.address == address:
            frequency_matrix[i, 0] += 1  # Increment the frequency count for the current address


def generate_distance_from_hub_clusters():
    # Calculate the distances of each location from the center point (address at index 0)
    center_distances = distances[0]

    # Create a scatter plot of the distances from the center point, coloring each point by its cluster
    plt.figure(figsize=(10, 6))
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        plt.scatter(cluster_indices, center_distances[cluster_indices], label=f'Cluster {cluster_id}', alpha=0.5)

    plt.xlabel('Location Index')
    plt.ylabel('Distance from Hub')
    plt.title('Distance from Hub Scatter Plot with Clusters')
    plt.legend()
    plt.show()


def generate_delivery_distribution():
    # Create a list to store the total number of deliveries for each address
    total_deliveries_per_address = [sum(routes_matrix[:, i]) for i in range(routes_matrix.shape[1])]

    # Create a histogram
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(raw_address_data)), total_deliveries_per_address, color='b', alpha=0.7)
    plt.xlabel('Address')
    plt.ylabel('Number of Deliveries')
    plt.title('Distribution of Completed Deliveries to Each Address')
    plt.xticks(range(len(raw_address_data)), [address[2] for address in raw_address_data], rotation=90, fontsize=6)
    plt.xlim(-2, 102)
    plt.tight_layout()
    plt.show()


def generate_total_route_duration():
    # Calculate the total route duration for each truck
    total_route_durations = [truck.accumulated_time.total_seconds() / 3600 for truck in trucks]

    # Create a bar chart
    plt.figure(figsize=(12, 6))
    plt.barh(range(1, len(trucks) + 1), total_route_durations, color='b', alpha=0.7)
    plt.xlabel('Total Route Duration (hours)')
    plt.ylabel('Truck ID')
    plt.title('Total Route Duration for Each Truck')
    plt.yticks(range(1, len(trucks) + 1))
    plt.tight_layout()
    plt.show()


def generate_route_duration_vs_iterations():
    plt.figure(figsize=(12, 6))

    for truck in trucks:
        # Calculate the total route duration for each iteration of simulated annealing
        total_route_durations = route_durations[truck.truck_id]
        iteration_counts = list(range(1, len(total_route_durations) + 1))
        # Convert the total route durations to a list of floats
        total_route_durations_float = [duration.total_seconds() / 3600 for duration in total_route_durations]
        # Plot the iteration count versus the total route duration for each truck
        plt.plot(iteration_counts, total_route_durations_float, marker='o', markersize=1,
                 label=f'Truck {truck.truck_id}')

    plt.xlabel('Number of Iterations of Simulated Annealing')
    plt.ylabel('Total Route Duration (hours)')
    plt.title('Total Route Duration vs. Number of Iterations of Simulated Annealing for Each Truck')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_visualizations_window():
    # Create a new window for visualizations
    visualizations_window = tk.Toplevel()
    visualizations_window.title("Visualizations")

    # Set the window size
    visualizations_window.geometry("500x200")

    # Add buttons to generate each visualization
    button1 = tk.Button(visualizations_window, text="Addresses Distance from Hub and Cluster",
                        command=generate_distance_from_hub_clusters)
    button1.pack(side=tk.TOP, pady=10)

    button2 = tk.Button(visualizations_window, text="Distribution of Completed Deliveries to Each Address",
                        command=generate_delivery_distribution)
    button2.pack(side=tk.TOP, pady=10)

    button3 = tk.Button(visualizations_window, text="Total Route Duration for Each Truck",
                        command=generate_total_route_duration)
    button3.pack(side=tk.TOP, pady=10)

    button4 = tk.Button(visualizations_window,
                        text="Total Route Duration vs. Number of Iterations of Simulated Annealing for Each Truck",
                        command=generate_route_duration_vs_iterations)
    button4.pack(side=tk.TOP, pady=10)


def visualize_data():
    # Call the function to create the visualizations window
    create_visualizations_window()


def create_truck_and_package_data_window():
    # Create a new window for displaying all packages information
    def view_package_status():
        user_time = time_entry.get()
        try:
            h, m = user_time.split(":")
            convert_timedelta = datetime.timedelta(hours=int(h), minutes=int(m))
        except ValueError:
            messagebox.showerror("Error", "Please enter time in HH:MM format.")
            return

        package_id = package_id_entry.get()
        if package_id.isdigit():
            package_id = int(package_id)
            if package_id not in range(1, 200):
                messagebox.showerror("Error", "Package not found. Please enter a valid package ID (1-40).")
                return
            else:
                package = package_table.get_package(package_id)
                display_package_info(package, convert_timedelta)
        elif package_id.lower() == "all":
            packages = package_table.get_all_packages()
            display_all_packages_info(packages, convert_timedelta)
        else:
            messagebox.showerror("Error", "Please enter a numeric package ID, 'all', or 'exit'.")
            return

    def display_package_info(package, convert_timedelta):
        package.set_converted_delivery_time(convert_timedelta)
        package.status_check(convert_timedelta)

        # Create a new window for displaying all packages information
        package_window = tk.Toplevel()
        package_window.title("Package Information")

        # Create a treeview widget with scrollbars
        tree = ttk.Treeview(package_window,
                            columns=("Truck ID", "Address", "Weight", "Delivery Time", "Status", "Loaded Time"))
        tree.heading("#0", text="Package ID", anchor=tk.CENTER)
        tree.column("#0", minwidth=0, width=100, stretch=tk.NO, anchor=tk.CENTER)  # Set the width of the first column
        tree.heading("Truck ID", text="Truck ID", anchor=tk.CENTER)
        tree.heading("Address", text="Address", anchor=tk.CENTER)
        tree.heading("Weight", text="Weight (lbs)", anchor=tk.CENTER)
        tree.heading("Delivery Time", text="Delivery Time", anchor=tk.CENTER)
        tree.heading("Status", text="Status", anchor=tk.CENTER)
        tree.heading("Loaded Time", text="Loaded Time", anchor=tk.CENTER)

        tree.insert("", "end", text=str(package.package_id),
                    values=(package.truck_info,
                            f"{package.address}, {package.city}, {package.state} {package.zip_code}",
                            package.weight, package.converted_delivery_time, package.status,
                            package.loaded_time))

        # Add the treeview and scrollbars to the window
        tree.pack(fill=tk.BOTH, expand=True)
        scroll_y = ttk.Scrollbar(package_window, orient=tk.VERTICAL, command=tree.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scroll_y.set)

        # Center align all columns
        for col in tree["columns"]:
            tree.heading(col, anchor=tk.CENTER)
            tree.column(col, anchor=tk.CENTER)

        # Set the minimum size of the window
        package_window.minsize(800, 400)

        # Start the main event loop for the window
        package_window.mainloop()

    def display_all_packages_info(packages, convert_timedelta):
        # Set the converted delivery time for each package
        for package in packages:
            package.set_converted_delivery_time(convert_timedelta)
            package.status_check(convert_timedelta)

        sorted_packages = sorted(packages, key=lambda x: int(x.package_id))

        # Create a new window for displaying all packages information
        all_packages_window = tk.Toplevel()
        all_packages_window.title("All Packages Information")

        # Create a treeview widget with scrollbars
        tree = ttk.Treeview(all_packages_window,
                            columns=(
                                "Truck ID", "Address", "Weight", "Delivery Time", "Status", "Loaded Time"))
        tree.heading("#0", text="Package ID", anchor=tk.CENTER)
        tree.column("#0", minwidth=0, width=100, stretch=tk.NO, anchor=tk.CENTER)  # Set the width of the first column
        tree.heading("Truck ID", text="Truck ID", anchor=tk.CENTER)
        tree.heading("Address", text="Address", anchor=tk.CENTER)
        tree.heading("Weight", text="Weight (lbs)", anchor=tk.CENTER)
        tree.heading("Delivery Time", text="Delivery Time", anchor=tk.CENTER)
        tree.heading("Status", text="Status", anchor=tk.CENTER)
        tree.heading("Loaded Time", text="Loaded Time", anchor=tk.CENTER)

        # Insert data into the treeview
        for package in sorted_packages:
            tree.insert("", "end", text=str(package.package_id),
                        values=(package.truck_info,
                                f"{package.address}, {package.city}, {package.state} {package.zip_code}",
                                package.weight, package.converted_delivery_time, package.status,
                                package.loaded_time))

        # Add the treeview and scrollbars to the window
        tree.pack(fill=tk.BOTH, expand=True)
        scroll_y = ttk.Scrollbar(all_packages_window, orient=tk.VERTICAL, command=tree.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scroll_y.set)

        # Center align all columns
        for col in tree["columns"]:
            tree.heading(col, anchor=tk.CENTER)
            tree.column(col, anchor=tk.CENTER)

        # Set the minimum size of the window
        all_packages_window.minsize(800, 400)

        # Start the main event loop for the window
        all_packages_window.mainloop()

    def view_mileage():
        # Create a new window for truck mileage
        mileage_window = tk.Toplevel()
        mileage_window.title("Truck Mileage")

        # Create a treeview widget with scrollbars
        tree = ttk.Treeview(mileage_window, columns=("Truck ID", "Mileage", "Route Time", "Departure Time"))
        tree.heading("#0", text="Truck ID", anchor=tk.CENTER)
        tree.column("#0", minwidth=0, width=100, stretch=tk.NO, anchor=tk.CENTER)  # Set the width of the first column
        tree.heading("Truck ID", text="Truck ID", anchor=tk.CENTER)
        tree.heading("Mileage", text="Mileage", anchor=tk.CENTER)
        tree.heading("Route Time", text="Route Time", anchor=tk.CENTER)
        tree.heading("Departure Time", text="Departure Time", anchor=tk.CENTER)

        # Insert data into the treeview
        for truck in trucks:
            tree.insert("", "end", text=str(truck.truck_id),
                        values=(
                            truck.truck_id, f"{truck.mileage:.2f} miles", truck.accumulated_time, truck.departure_time))

        # Add the treeview and scrollbars to the window
        tree.pack(fill=tk.BOTH, expand=True)
        scroll_y = ttk.Scrollbar(mileage_window, orient=tk.VERTICAL, command=tree.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scroll_y.set)

        # Center align all columns
        for col in tree["columns"]:
            tree.heading(col, anchor=tk.CENTER)
            tree.column(col, anchor=tk.CENTER)

        total_route_time = 0
        for truck in trucks:
            total_route_time += truck.accumulated_time.total_seconds()

        average_route_time = total_route_time / len(trucks)
        print(f'{average_route_time}')

        # Start the main event loop for the window
        mileage_window.mainloop()

    # Create a new window for truck and package data
    data_window = tk.Toplevel()
    data_window.title("Truck and Package Data")
    data_window.geometry("300x200")

    # Create widgets
    time_label = tk.Label(data_window, text="Enter time (HH:MM):")
    time_entry = tk.Entry(data_window)

    package_id_label = tk.Label(data_window, text="Enter package ID or 'all':")
    package_id_entry = tk.Entry(data_window)

    package_status_button = tk.Button(data_window, text="Check Package Status", command=view_package_status)
    mileage_button = tk.Button(data_window, text="View Truck Mileage", command=view_mileage)

    mileage_display = tk.Label(data_window, text="")

    # Place widgets in the window
    time_label.grid(row=0, column=0, padx=10, pady=10)
    time_entry.grid(row=0, column=1, padx=10, pady=10)
    package_id_label.grid(row=1, column=0, padx=10, pady=10)
    package_id_entry.grid(row=1, column=1, padx=10, pady=10)
    package_status_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
    mileage_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)
    mileage_display.grid(row=4, column=0, columnspan=2, padx=10, pady=10)


def truck_and_package_data():
    # Call the function to create the visualizations window
    create_truck_and_package_data_window()


def main():
    # Create the main window
    root = tk.Tk()
    root.title("Main Menu")
    root.geometry("300x200")
    label = tk.Label(root, text="Select an option:")
    label.pack()
    button1 = tk.Button(root, text="Visualizations", command=visualize_data)
    button1.pack(side=tk.TOP, pady=10)
    button2 = tk.Button(root, text="Truck and Package Data", command=truck_and_package_data)
    button2.pack(side=tk.TOP, pady=10)
    exit_button = tk.Button(root, text="Exit", command=root.destroy)
    exit_button.pack(side=tk.TOP, pady=10)
    root.mainloop()


if __name__ == "__main__":
    main()
