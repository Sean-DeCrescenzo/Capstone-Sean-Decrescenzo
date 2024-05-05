# Custom Package class

class Package:
    def __init__(self, package_id, address, city, state, zip_code, weight, status, delivery_time, loaded_time,
                 truck_info):
        self.package_id = package_id
        self.address = address
        self.city = city
        self.state = state
        self.zip_code = zip_code
        self.weight = weight
        self.status = status
        self.delivery_time = delivery_time
        self.loaded_time = loaded_time
        self.truck_info = str(truck_info)

    # Print function for package information with logic to determine delivery time value
    def __str__(self):
        loaded_time_str = str(self.loaded_time)
        if self.status == "At Hub" or self.status == "En route":
            delivery_time_temp = "Pending"
        else:
            delivery_time_temp = str(self.delivery_time) + ' AM'
        return (f"Package ID: {self.package_id:<2} "
                f"| TruckId: {self.truck_info:<2} "
                f"| Address: {self.address + ', ' + self.city + ', ' + self.state + ' ' + self.zip_code:<66} "
                f"| Weight (lbs): {self.weight:<2} "
                f"| Delivery Time: {delivery_time_temp:<10} "
                f"| Status: {self.status:<9} "
                f"| Loaded Time: {loaded_time_str:<7} ")

