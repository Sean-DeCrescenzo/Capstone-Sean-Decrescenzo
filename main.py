# Student Name: Sean Decrescenzo
# Student ID: 000973102

# Import necessary modules and classes
import csv
import datetime
import math
import os
import sys
import truck
from package import Package
from hashTable import HashTable
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import random
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


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
        root.destroy()  # Close the Tkinter window

# file_path = 'CSV/WGUPSPackageFile2.0.csv'
# load_package_data(file_path, package_table)

root = tk.Tk()
root.title("File Selection Example")

button = tk.Button(root, text="Select File", command=select_file)
button.pack(padx=40, pady=40)

root.mainloop()

# Initialize a list to store the delivered packages
trucks = []

for i in range(1, 5):
    new_truck = truck.Truck(i, i, [], 0.0, "4001 South 700 East", 30, datetime.timedelta(hours=7), 0)
    trucks.append(new_truck)


def load_packages(truck, all_packages_off_trucks, distance_data, total_weight_all_packages, total_trucks):
    current_weight = truck.load_weight  # Start with the current load weight of the truck
    average_weight_per_truck = total_weight_all_packages // len(total_trucks)

    while current_weight < average_weight_per_truck and current_weight < truck.max_weight:
        # Select a random package from the remaining packages
        random_package = random.choice(all_packages_off_trucks)
        package_weight = random_package.weight

        # Check if the package's weight can be accommodated
        if current_weight + package_weight <= average_weight_per_truck:
            # Add the package to the truck
            truck.packages.append(random_package.package_id)
            current_weight += package_weight
            truck.load_weight += package_weight  # Update the truck's load weight
            all_packages_off_trucks.remove(random_package)
        else:
            break

    # If there are still packages left and this is the last truck, add them to this truck
    if len(all_packages_off_trucks) > 0 and truck == total_trucks[-1]:
        for package in all_packages_off_trucks:
            if current_weight + package.weight <= truck.max_weight:
                truck.packages.append(package)
                current_weight += package.weight
                truck.load_weight += package.weight
                all_packages_off_trucks.remove(package)


all_packages_off_trucks = package_table.get_all_packages()
total_weight_all_packages = sum(package.weight for package in all_packages_off_trucks)

undelivered_packages = {}
delivery_order = {}
route_durations = {}
# Assuming total_trucks is the total number of trucks
for truck in trucks:
    load_packages(truck, all_packages_off_trucks, raw_distance_data, total_weight_all_packages, trucks)
    undelivered_packages[truck.truck_id] = truck.packages
    delivery_order[truck.truck_id] = []
    route_durations[truck.truck_id] = []


def calculate_distance(address1, address2, distance_data):
    index1 = get_destination_index(address1)
    index2 = get_destination_index(address2)
    if index1 is not None and index2 is not None:
        distance = distance_data[index1][index2]
        if distance == '':
            distance = distance_data[index2][index1]
        return float(distance)
    return float('inf')


def get_destination_index(address):
    for i, record in enumerate(raw_address_data):
        if record[2] == address:
            return i
    return None


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
    cooling_rate = 0.0005
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
            arrival_datetime_at_hub = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + arrival_time_at_hub

            print(
                f"Truck {current_truck.truck_id} - Arrival Time at Hub: {arrival_datetime_at_hub.strftime('%I:%M %p')}")


# Deliver packages
for truck in trucks:
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


def generate_delivery_routes_heatmap():
    # Calculate the distances from each address to the hub
    hub_address = raw_address_data[0][2]
    distances_to_hub = [calculate_distance(hub_address, address[2], raw_distance_data) for address in raw_address_data]

    # Create a grid of coordinates for the heatmap
    x_coords = np.linspace(-25, 25, 101)
    y_coords = np.linspace(-20, 20, 101)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Calculate the distance of each point in the grid from the hub
    distances_from_hub = np.sqrt(X ** 2 + Y ** 2)

    # Create a heatmap based on the number of packages delivered to each location
    heatmap = np.zeros_like(distances_from_hub)
    for i, distance in enumerate(distances_to_hub):
        heatmap += np.where(np.isclose(distances_from_hub, distance, atol=0.5), frequency_matrix[i], 0)

    # Plot the heatmap
    plt.figure(figsize=(11, 7))
    plt.imshow(heatmap, cmap='viridis', origin='lower', extent=(-25, 25, -20, 20), alpha=1, interpolation='spline36')
    plt.colorbar(label='Number of Deliveries', shrink=0.5)

    # Plot each address as a point on the heatmap
    hub_x, hub_y = 0, 0  # Position of the hub
    plt.scatter([hub_x], [hub_y], c='blue', label='Hub', s=100)  # Plot the hub at the center
    for i, distance in enumerate(distances_to_hub[1:], start=1):
        angle = i * (2 * np.pi / len(distances_to_hub[1:]))  # Calculate the angle for positioning
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        plt.scatter(x, y, c='white')  # Plot each address based on its distance and angle

    # Create a legend for the hub and addresses
    legend_elements = [
        plt.Line2D([0], [0], marker='o', markerfacecolor='blue', markersize=10, label='Hub', linestyle='None'),
        plt.Line2D([0], [0], marker='o', markerfacecolor='white', markersize=10, label='Addresses', linestyle='None')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.xlabel('Distance from Hub', labelpad=10)
    plt.ylabel('Distance from Hub', labelpad=10)
    plt.title('Delivery Routes Heat Map')
    plt.tight_layout(pad=0.2)
    plt.show(block=True)


def generate_delivery_distribution():
    # Create a list to store the total number of deliveries for each address
    total_deliveries_per_address = [sum(routes_matrix[:, i]) for i in range(routes_matrix.shape[1])]

    # Create a histogram
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(raw_address_data)), total_deliveries_per_address, color='b', alpha=0.7)
    plt.xlabel('Address Index')
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

    # Add a label
    label = tk.Label(visualizations_window, text="Delivery Route Visualizations")
    label.pack()

    # Add buttons to generate each visualization
    button1 = tk.Button(visualizations_window, text="Delivery Routes Heat Map",
                        command=generate_delivery_routes_heatmap)
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
