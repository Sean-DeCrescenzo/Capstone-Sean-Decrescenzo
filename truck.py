# Custom Truck class
import datetime

class Truck:
    # Class attribute for max weight
    max_weight = 1500

    def __init__(self, truck_id, driver, packages, mileage, location, speed, departure_time, load_weight):
        self.truck_id = truck_id
        self.driver = driver
        self.packages = packages  # Store package IDs as a separate attribute
        self.mileage = mileage
        self.location = location
        self.speed = speed
        self.departure_time = departure_time
        self.accumulated_time = datetime.timedelta()  # Initialize accumulated time to zero
        self.load_weight = load_weight

    # Print function for truck data
    def __str__(self):
        return f"Truck ID: {self.truck_id}, " \
               f"Driver: {self.driver}, " \
               f"Maximum Load: {self.max_load}, " \
               f"Packages: {self.packages}, " \
               f"Mileage: {self.mileage}, " \
               f"Location: {self.location}, " \
               f"Speed: {self.speed} mph, " \
               f"Departure Time: {self.departure_time}"
