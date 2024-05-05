# Student Name: Sean Decrescenzo
# Student ID: 000973102
import csv
import datetime
import math
import os
import sys
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import RadiusNeighborsTransformer
import truck
from package import Package
from hashTable import HashTable

logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

print('Student Name: Sean Decrescenzo\n'
      'Student ID:   000973102')

# Retrieve base path of program for .exe file.
if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

csv_folder_path = os.path.join(base_path, 'CSV')
if not os.path.exists(csv_folder_path):
    base_path = os.path.dirname(base_path)
    csv_folder_path = os.path.join(base_path, 'CSV')

# Address and Distance matrix data filepath.
csv_path_distance = os.path.join(csv_folder_path, 'WGUPSDistanceFile2.0.csv')
csv_path_address = os.path.join(csv_folder_path, 'WGUPSAddressFile2.0.csv')

with open(csv_path_distance, 'r') as csvDistanceTable:
    raw_distance_data = list(csv.reader(csvDistanceTable))
with open(csv_path_address, 'r') as csvAddressTable:
    raw_address_data = list(csv.reader(csvAddressTable))

# Initialize package_table from HashTable class to store package data
package_table = HashTable(10)


# Load package data into hash table
def load_package_data(file_path, package_data_table):
    with open(file_path, 'r') as csv_package_file:
        csv_package_data = csv.reader(csv_package_file)
        next(csv_package_data)
        for row in csv_package_data:
            if len(row) >= 6:
                p_id = row[0].strip()
                address = row[1].strip()
                city = row[2].strip()
                state = row[3].strip()
                zip_code = row[4].strip()
                weight = int(row[5].strip())

                if not (p_id and address and city and state and zip_code and weight):
                    print(f"Ignoring invalid row: {row}")
                    continue

                imported_package = Package(
                    p_id, address, city, state, zip_code,
                    weight, "At Hub", None,
                    None, None)
                package_data_table.insert(int(imported_package.package_id), imported_package)
            else:
                print(f"Ignoring invalid row: {row}")


# User file selection UI.
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        print(f"Selected package file: {file_path}")
        load_package_data(file_path, package_table)
        root.destroy()


root = tk.Tk()
root.title("WGUPS Logistics Tool")
button = tk.Button(root, text="Please Select Package File", command=select_file)
button.pack(padx=40, pady=40)
root.minsize(400, 200)
root.mainloop()

# Initialize 4 delivery trucks and associated elements
trucks = []
for i in range(0, 4):
    new_truck = truck.Truck(
        i, i, [], 0.0, "4001 South 700 East", 30,
        datetime.timedelta(hours=7), 0
    )
    trucks.append(new_truck)


# Method to retrieve the distance between two addresses from the distance data matrix
def calculate_distance(address1, address2, distance_data):
    index1 = get_destination_index(address1)
    index2 = get_destination_index(address2)
    if index1 is not None and index2 is not None:
        distance_between_indexes = distance_data[index1][index2]
        if distance_between_indexes == '':
            distance_between_indexes = distance_data[index2][index1]
        return float(distance_between_indexes)
    return float('inf')


# Retrieves the index number for a given street address.
def get_destination_index(street_address):
    for address_row, record in enumerate(raw_address_data):
        if record[2] == street_address:
            return address_row
    return None


# Gather a list of all street addresses to create distance matrix
address_list = [record[2] for record in raw_address_data]

# Calculate the distances between addresses
distances = np.zeros((len(address_list), len(address_list)))
for i, address_i in enumerate(address_list):
    for j, address_j in enumerate(address_list[i + 1:], start=i + 1):
        distance = calculate_distance(address_i, address_j, raw_distance_data)
        distances[i, j] = distance
        distances[j, i] = distance

# Create a RadiusNeighborsTransformer object and map neighbors as indexes within 10.0 distance
rnt = RadiusNeighborsTransformer(radius=10, mode='distance', metric='precomputed')

# Fit the transformer to the distances
rnt.fit(distances)

# Initialize an empty connectivity matrix the size of the raw distance data as array of zeros.
connectivity_matrix = np.zeros((len(address_list), len(address_list)))

# Get the nearest neighbors for a given index using the rnt.
# Then add the data to the connectivity_matrix array.
for i in range(len(address_list)):
    neighbors_indices = rnt.radius_neighbors([distances[i]], return_distance=False)[0]
    nearest_neighbors = [address_list[i] for i in neighbors_indices]
    connectivity_matrix[i, neighbors_indices] = 1

# Cluster the addresses into four clusters using the connectivity_matrix array and distances array.
clusterer = AgglomerativeClustering(n_clusters=4, connectivity=connectivity_matrix, linkage='ward')
clusters = clusterer.fit_predict(distances)

# Assign packages to clusters based on the assigned cluster of their delivery address.
package_clusters = {}
for package_id, package in package_table.items():
    cluster_id = clusters[get_destination_index(package.address)]
    if cluster_id not in package_clusters:
        package_clusters[cluster_id] = []
    package_clusters[cluster_id].append((package_id, package))

# Load the trucks with packages from the cluster with the same cluster_id as its truck_id
for truck in trucks:
    cluster_id = truck.truck_id
    if cluster_id in package_clusters:
        cluster = package_clusters[cluster_id]
        while cluster and truck.load_weight < truck.max_weight:
            package_id, package = cluster[0]
            if package.weight <= truck.max_weight - truck.load_weight:
                truck.packages.append(str(package_id))
                truck.load_weight += package.weight
                cluster.pop(0)
            else:
                break
        if not cluster:
            del package_clusters[cluster_id]

        # Check if there is still space in the truck
        # Fill any remaining space in the trucks with remaining packages in clusters
        if truck.load_weight < truck.max_weight:
            for c_id, cluster in package_clusters.items():
                while cluster and truck.load_weight < truck.max_weight:
                    package_id, package = cluster[0]
                    if package.weight <= truck.max_weight - truck.load_weight:
                        truck.packages.append(str(package_id))
                        truck.load_weight += package.weight
                        cluster.pop(0)
                    else:
                        break
                if not cluster:
                    del package_clusters[c_id]


# Calculates total route time for a truck.
def calculate_route_time(current_truck, truck_delivery_order, distance_data):
    current_location = "4001 South 700 East"
    total_time = datetime.timedelta()
    for next_package in truck_delivery_order:
        current_package = package_table.get_package(int(next_package))
        delivery_distance = calculate_distance(current_location, current_package.address, distance_data)
        delivery_time = datetime.timedelta(hours=delivery_distance / current_truck.speed)
        total_time += delivery_time
        current_location = current_package.address
    return total_time


# Initialize required objects for simulated annealing method.
delivery_order = {}
route_durations = {}


# Simulated annealing method for optimizing the delivery route of a truck.
# Uses a simulated annealing algorithm to find the best order of delivery for packages on the truck.
# Inputs:
#   - current_truck: The truck object representing the current state of the truck.
#   - distance_data: The distance data used to calculate route times.
# Returns:
#   The best order of delivery for packages on the truck after optimization.
def simulated_annealing_truck_route(current_truck, distance_data):
    current_order = current_truck.packages
    if len(current_order) < 2:
        return current_order  # Skip optimization if there are less than 2 packages
    original_time = calculate_route_time(current_truck, current_order, distance_data)
    current_time = calculate_route_time(current_truck, current_order, distance_data)
    best_order = current_order.copy()
    best_time = current_time
    temperature = 221.25
    cooling_rate = 0.002
    min_temperature = 0.01
    iteration_count = 0
    while temperature > min_temperature:
        new_order = current_order.copy()
        if len(new_order) < 2:
            continue  # Skip this iteration if there are less than 2 packages
        package_index1, package_index2 = random.sample(range(len(new_order)), 2)
        new_order[package_index1], new_order[package_index2] = new_order[package_index2], new_order[package_index1]
        new_time = calculate_route_time(current_truck, new_order, distance_data)
        cost_diff = new_time - current_time
        if (cost_diff.total_seconds() < 0 or
                random.random() < math.exp(-cost_diff.total_seconds() / temperature)):
            current_order = new_order.copy()
            current_time = new_time
            if current_time < best_time:
                best_order = current_order.copy()
                best_time = current_time
        temperature *= 1 - cooling_rate
        iteration_count += 1
        # Save the total route duration for this iteration
        route_durations[current_truck.truck_id].append(best_time)

    print(f"New Route Time: {best_time} \n"
          f"OriginalTime: {original_time} \n"
          f"Iterations for Truck {current_truck.truck_id}: {iteration_count}\n"
          f"packages on Truck {len(current_truck.packages)}\n")
    return best_order






# Optimize routes for each truck using simulated annealing method
for truck in trucks:
    route_durations[truck.truck_id] = []
    truck.packages = simulated_annealing_truck_route(truck, raw_distance_data)


# Delivers the next package in the truck's package list and adds distance traveled to the truck's mileage.
def deliver_package(current_truck, next_package, delivery_distance):
    delivery_time = datetime.timedelta(hours=delivery_distance / current_truck.speed)
    current_truck.accumulated_time += delivery_time
    next_package.loaded_time = current_truck.departure_time
    next_package.truck_info = current_truck.truck_id
    next_package.delivery_time = current_truck.departure_time + current_truck.accumulated_time
    current_truck.mileage += delivery_distance
    current_truck.location = next_package.address


# Method to deliver all packages on each truck while tracking various data.
def deliver_packages(current_truck, distance_data):
    while truck.packages:
        next_package = package_table.get_package(int(truck.packages[0]))
        if next_package is not None:
            delivery_distance = calculate_distance(truck.location, next_package.address, distance_data)
            deliver_package(current_truck, next_package, delivery_distance)
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
            current_truck.accumulated_time += datetime.timedelta(
                hours=distance_to_hub / current_truck.speed)
            current_truck.location = "4001 South 700 East"
            # Track the time of arrival at the hub
            arrival_time_at_hub = current_truck.departure_time + current_truck.accumulated_time
            arrival_datetime_at_hub = datetime.datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0) + arrival_time_at_hub

            print(
                f"Truck {current_truck.truck_id} - "
                f"Arrival Time at Hub: {arrival_datetime_at_hub.strftime('%I:%M %p')}")


# Iterate over all the trucks and deliver packages for each truck.
# Track the order in which each truck delivered packages.
for truck in trucks:
    if not truck.packages:
        continue  # Skip trucks with no packages
    delivery_order[truck.truck_id] = []
    deliver_packages(truck, raw_distance_data)

# Create a matrix to represent the delivery routes
# Then fill the matrix with the delivery order for each truck
routes_matrix = np.zeros((len(trucks), len(raw_address_data)))
for truck_id, delivery_order in delivery_order.items():
    for package_id in delivery_order:
        address_index = get_destination_index(package_table.get_package(int(package_id)).address)
        routes_matrix[truck_id - 1][address_index] += 1


# Visualization for each locations distance from the delivery hub.
# Each plotted location's cluster is indicated by its plotted points color.
def generate_distance_from_hub_clusters():
    # Calculate the distances of each location from the center point (address at index 0)
    center_distances = distances[0]

    plt.figure(figsize=(10, 6))
    for current_cluster in np.unique(clusters):
        cluster_indices = np.where(clusters == current_cluster)[0]
        plt.scatter(
            cluster_indices,
            center_distances[cluster_indices],
            label=f'Cluster {current_cluster}',
            alpha=0.5)

    plt.xlabel('Location Index')
    plt.ylabel('Distance from Hub')
    plt.title('Distance from Hub Scatter Plot with Clusters')
    plt.legend()
    plt.show()


# Visualization of total number of deliveries for each address using the routes_matrix.
def generate_delivery_distribution():
    total_deliveries_per_address = [sum(routes_matrix[:, delivered_package])
                                    for delivered_package in range(routes_matrix.shape[1])]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(raw_address_data)), total_deliveries_per_address, color='b', alpha=0.7)
    plt.xlabel('Address')
    plt.ylabel('Number of Deliveries')
    plt.title('Distribution of Completed Deliveries to Each Address')
    plt.xticks(range(len(raw_address_data)), [address[2] for address in raw_address_data], rotation=90, fontsize=6)
    plt.xlim(-2, 102)
    plt.tight_layout()
    plt.show()


# Visualization bar chart of each trucks total route duration.
def generate_total_route_duration():
    total_route_durations = [current_truck.accumulated_time.total_seconds() / 3600 for current_truck in trucks]

    plt.figure(figsize=(12, 6))
    plt.barh(range(1, len(trucks) + 1), total_route_durations, color='b', alpha=0.7)
    plt.xlabel('Total Route Duration (hours)')
    plt.ylabel('Truck ID')
    plt.title('Total Route Duration for Each Truck')
    plt.yticks(range(1, len(trucks) + 1))
    plt.xticks(range(0, int(max(total_route_durations)) + 1))
    plt.tight_layout()
    plt.show()


def generate_route_duration_vs_iterations():
    plt.figure(figsize=(12, 6))

    for current_truck in trucks:
        # Calculate the total route duration for each iteration of simulated annealing
        total_route_durations = route_durations[current_truck.truck_id]
        iteration_counts = list(range(1, len(total_route_durations) + 1))
        # Convert the total route durations to a list of floats
        total_route_durations_float = [duration.total_seconds() / 3600 for duration in total_route_durations]
        # Plot the iteration count versus the total route duration for each truck
        plt.plot(iteration_counts, total_route_durations_float, marker='o', markersize=1,
                 label=f'Truck {current_truck.truck_id}')

    plt.xlabel('Number of Iterations of Simulated Annealing')
    plt.ylabel('Total Route Duration (hours)')
    plt.title('Total Route Duration vs. Number of Iterations of Simulated Annealing for Each Truck')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_visualizations_window():
    visualizations_window = tk.Toplevel()
    visualizations_window.title("Visualizations")
    visualizations_window.geometry("500x200")

    button1 = tk.Button(visualizations_window, text="Addresses Distance from Hub and Locale Cluster",
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
    create_visualizations_window()


def create_truck_and_package_data_window():
    data_window = tk.Toplevel()
    data_window.title("Truck and Package Data")
    data_window.geometry("800x400")

    # Truck Information Section
    truck_frame = tk.Frame(data_window, padx=10, pady=10)
    truck_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    truck_label = tk.Label(truck_frame, text="Truck Information", font=("Arial", 12, "bold"))
    truck_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

    truck_id_label = tk.Label(truck_frame, text="Enter Truck ID:")
    truck_id_label.grid(row=1, column=0, padx=10, pady=10)
    truck_id_entry = tk.Entry(truck_frame)
    truck_id_entry.grid(row=1, column=1, padx=10, pady=10)
    truck_load_view_button = tk.Button(truck_frame, text="View Optimal Truck Load",
                                       command=lambda: view_truck_packages(truck_id_entry.get()))
    truck_load_view_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    mileage_button = tk.Button(truck_frame, text="View All Truck Route Details", command=view_mileage)
    mileage_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    # Package Information Section
    package_frame = tk.Frame(data_window, padx=10, pady=10)
    package_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
    package_label = tk.Label(package_frame, text="Package Information", font=("Arial", 12, "bold"))
    package_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

    package_id_label = tk.Label(package_frame, text="Enter package ID or 'all':")
    package_id_label.grid(row=1, column=0, padx=10, pady=10)
    package_id_entry = tk.Entry(package_frame)
    package_id_entry.grid(row=1, column=1, padx=10, pady=10)
    package_status_button = tk.Button(package_frame, text="Check Package Information",
                                      command=lambda: view_package_status(package_id_entry.get()))
    package_status_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    data_window.rowconfigure(0, weight=1)
    data_window.columnconfigure(0, weight=1)
    data_window.columnconfigure(1, weight=1)
    data_window.mainloop()


def view_package_status(package_id_entry):
    user_package_id = package_id_entry
    if user_package_id.isdigit():
        user_package_id = int(user_package_id)
        if user_package_id not in range(1, 500):
            messagebox.showerror("Error", "Package not found. Please enter a valid package ID (1-40).")
            return
        user_package = package_table.get_package(user_package_id)
        display_package_info(user_package)
    elif user_package_id.lower() == "all":
        packages = package_table.get_all_packages()
        display_all_packages_info(packages, display_truck_id=None)
    else:
        messagebox.showerror("Error", "Please enter a numeric package ID, 'all', or 'exit'.")
        return


def display_package_info(package_data):
    package_window = tk.Toplevel()
    package_window.title("Package Information")

    tree = ttk.Treeview(package_window,
                        columns=("Truck ID", "Address", "Weight", "Delivery Time", "Loaded Time"))
    tree.heading("#0", text="Package ID", anchor=tk.CENTER)
    tree.column("#0", minwidth=0, width=100, stretch=tk.NO, anchor=tk.CENTER)
    tree.heading("Truck ID", text="Truck ID", anchor=tk.CENTER)
    tree.heading("Address", text="Address", anchor=tk.CENTER)
    tree.heading("Weight", text="Weight (lbs)", anchor=tk.CENTER)
    tree.heading("Delivery Time", text="Estimated Delivery Time", anchor=tk.CENTER)
    tree.heading("Loaded Time", text="Loaded Time", anchor=tk.CENTER)

    package_data.converted_delivery_time = package.delivery_time
    truck_info = int(package_data.truck_info) + 1 if package_data.truck_info != 'None' else "Not on truck"
    tree.insert("", "end", text=str(package_data.package_id),
                values=(truck_info,
                        f"{package_data.address}, {package_data.city}, {package_data.state} "
                        f"{package_data.zip_code}", package_data.weight, package_data.converted_delivery_time,
                        package_data.loaded_time))

    tree.pack(fill=tk.BOTH, expand=True)
    scroll_y = ttk.Scrollbar(package_window, orient=tk.VERTICAL, command=tree.yview)
    scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scroll_y.set)

    for col in tree["columns"]:
        tree.heading(col, anchor=tk.CENTER)
        tree.column(col, anchor=tk.CENTER)
    package_window.minsize(800, 400)
    package_window.mainloop()


def display_all_packages_info(packages, display_truck_id):
    for package_data in packages:
        package_data.converted_delivery_time = package_data.delivery_time

    all_packages_window = tk.Toplevel()
    sorted_packages = []
    if display_truck_id is None:
        all_packages_window.title("All Packages Information")
        sorted_packages = sorted(packages, key=lambda x: int(x.package_id))
    if display_truck_id:
        all_packages_window.title(
            f"Optimal Delivery Order for Truck {display_truck_id}: All Estimates Based On 7:00AM Load Time")
        sorted_packages = sorted(packages, key=lambda x: x.delivery_time)

    tree = ttk.Treeview(all_packages_window,
                        columns=(
                            "Truck ID", "Address", "Weight", "Delivery Time", "Loaded Time"))
    tree.heading("#0", text="Package ID", anchor=tk.CENTER)
    tree.column("#0", minwidth=0, width=100, stretch=tk.NO, anchor=tk.CENTER)
    tree.heading("Truck ID", text="Truck ID", anchor=tk.CENTER)
    tree.heading("Address", text="Address", anchor=tk.CENTER)
    tree.heading("Weight", text="Weight (lbs)", anchor=tk.CENTER)
    tree.heading("Delivery Time", text="Estimated Delivery Time", anchor=tk.CENTER)
    tree.heading("Loaded Time", text="Loaded Time", anchor=tk.CENTER)

    for package_data in sorted_packages:
        truck_info = int(package_data.truck_info) + 1 if package_data.truck_info != 'None' else "Not on truck"
        loaded_time_str = str(package_data.loaded_time) if package_data.loaded_time else "Not loaded"
        tree.insert("", "end", text=str(package_data.package_id),
                    values=(
                        truck_info,
                        f"{package_data.address}, {package_data.city}, {package_data.state} "
                        f"{package_data.zip_code}", package_data.weight, package_data.converted_delivery_time,
                        loaded_time_str))

    tree.pack(fill=tk.BOTH, expand=True)
    scroll_y = ttk.Scrollbar(all_packages_window, orient=tk.VERTICAL, command=tree.yview)
    scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scroll_y.set)

    for col in tree["columns"]:
        tree.heading(col, anchor=tk.CENTER)
        tree.column(col, anchor=tk.CENTER)
    all_packages_window.minsize(800, 400)
    all_packages_window.mainloop()


def view_truck_packages(truck_id_entry):
    all_packages = package_table.get_all_packages()
    truck_packages = []

    corrected_truck_id = str(int(truck_id_entry) - 1)
    for current_package in all_packages:
        if str(current_package.truck_info) == str(corrected_truck_id):
            truck_packages.append(current_package)

    # Display all packages for the specified truck
    display_all_packages_info(truck_packages, truck_id_entry)


def view_mileage():
    mileage_window = tk.Toplevel()
    mileage_window.title("Truck Mileage")

    tree = ttk.Treeview(mileage_window, columns=("Mileage", "Route Time", "Departure Time"))
    tree.heading("#0", text="Truck ID", anchor=tk.CENTER)
    tree.column("#0", minwidth=0, width=100, stretch=tk.NO, anchor=tk.CENTER)
    tree.heading("Mileage", text="Mileage", anchor=tk.CENTER)
    tree.heading("Route Time", text="Route Time", anchor=tk.CENTER)
    tree.heading("Departure Time", text="Departure Time", anchor=tk.CENTER)

    for truck_data in trucks:
        tree.insert("", "end", text=str(truck_data.truck_id + 1),
                    values=(
                        f"{truck_data.mileage:.2f} miles", f"{truck_data.accumulated_time} hours",
                        f"{truck_data.departure_time} AM"))

    tree.pack(fill=tk.BOTH, expand=True)
    scroll_y = ttk.Scrollbar(mileage_window, orient=tk.VERTICAL, command=tree.yview)
    scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scroll_y.set)

    for col in tree["columns"]:
        tree.heading(col, anchor=tk.CENTER)
        tree.column(col, anchor=tk.CENTER)

    mileage_window.mainloop()


def truck_and_package_data():
    create_truck_and_package_data_window()


def main():
    information_menu = tk.Tk()
    information_menu.title("Main Menu")
    information_menu.geometry("300x200")
    label = tk.Label(information_menu, text="Select an option:")
    label.pack()
    button1 = tk.Button(information_menu, text="Visualizations", command=visualize_data)
    button1.pack(side=tk.TOP, pady=10)
    button2 = tk.Button(information_menu, text="Truck and Package Data", command=truck_and_package_data)
    button2.pack(side=tk.TOP, pady=10)
    exit_button = tk.Button(information_menu, text="Exit", command=information_menu.destroy)
    exit_button.pack(side=tk.TOP, pady=10)
    information_menu.mainloop()


if __name__ == "__main__":
    main()
