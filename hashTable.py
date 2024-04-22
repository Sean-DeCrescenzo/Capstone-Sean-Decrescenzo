# Custom class for Hash Table
class HashTable:
    # Initialize the HashTable with a given size.
    # This creates a list the length of the size of empty buckets.
    # Time complexity: O(1)
    # Space complexity: O(1)
    def __init__(self, size):
        self.size = size
        self.table = []
        for i in range(size):
            self.table.append([])

    # Hash function that returns the bucket dependent on the input value.
    # Time complexity: O(1)
    # Space complexity: O(1)
    def _hash(self, key):
        return int(key) % self.size

    # Insert a key-value pair into the HashTable.
    # If the key already exists, its value is updated.
    # If the key doesn't exist, a new key-value pair is added to the appropriate bucket.
    # Time complexity: O(N)
    # Space complexity: O(1)
    def insert(self, key, value):
        bucket = self._hash(key)
        bucket_list = self.table[bucket]
        for pair in bucket_list:
            if pair[0] == key:   # Check if the key already exists in the bucket by iterating through existing keys
                pair[1] = value  # Update the value if the key is found
                return
        key_value = [key, value]
        bucket_list.append(key_value)  # Add a new key-value pair if the key is not found
        return True

    # Retrieve the value associated with a key from the HashTable.
    # If the key is found in the existing keys the associated value is returned
    # Time complexity: O(N)
    # Space complexity: O(1)
    def get_package(self, key):
        bucket = self._hash(key)
        bucket_list = self.table[bucket]
        for pair in bucket_list:
            if pair[0] == key:
                return pair[1]  # Return the value if the key is found
        # If the key is not found, return an Error
        raise KeyError(f"Key '{key}' not found in the hash table")

    # Retrieve all values stored in the HashTable.
    # Time complexity: O(N)
    # Space complexity: O(N)
    def get_all_packages(self):
        all_values = []
        for bucket_list in self.table:
            for pair in bucket_list:
                all_values.append(pair[1])  # Append the value to the list of all values
        return all_values

    # Remove a key-value pair from the HashTable.
    # If the provided key is found in existing keys the key and its associated value are removed.
    # Time complexity: O(N)
    # Space complexity: O(1)
    def remove(self, key):
        bucket = self._hash(key)
        bucket_list = self.table[bucket]
        for pair in bucket_list:
            if pair[0] == key:
                bucket_list.remove(pair)  # Remove the key-value pair if the key is found
                return
        # If the key is not found, return an Error
        raise KeyError(f"Key '{key}' not found in the hash table")
