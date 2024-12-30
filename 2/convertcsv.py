import csv

# Input and output file names
input_file = "massif.out.8534"  # Replace with your actual file name
output_file = "example6_output.csv"

# Initialize storage for parsed data
data = []

# Parse the massif output file
with open(input_file, "r") as f:
    snapshot = {}
    for line in f:
        line = line.strip()
        if line.startswith("snapshot="):
            if snapshot:
                data.append(snapshot)
            snapshot = {}
        elif "=" in line:
            key, value = line.split("=")
            snapshot[key] = value
    if snapshot:
        data.append(snapshot)

# Write to CSV
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(["Time (ms)", "Memory Allocated (KB)", "Memory Freed (KB)", "Peak Memory (KB)", "Stack Usage (KB)"])
    # Write rows
    for snapshot in data:
        time = int(snapshot.get("time", 0))
        heap = int(snapshot.get("mem_heap_B", 0)) / 1024
        heap_extra = int(snapshot.get("mem_heap_extra_B", 0)) / 1024
        stack = int(snapshot.get("mem_stacks_B", 0)) / 1024
        peak_memory = heap + heap_extra
        writer.writerow([time, heap, 0, peak_memory, stack])

print(f"CSV written to {output_file}")
