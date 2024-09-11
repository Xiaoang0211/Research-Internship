import matplotlib.pyplot as plt

# Define the data and color maps
data = [3.92, 0.03, 0.03, 0.16, 0.20, 0.07, 0.07,0.05, 15.30, 1.12, 11.13, 0.56,  14.10, 3.90, 39.30,  0.51, 9.17, 0.29,  0.08]
color_map = {
    1: [245, 150, 100],
    2: [245, 230, 100],
    3: [150, 60, 30],
    4: [180, 30, 80],
    5: [255, 0, 0],
    6: [30, 30, 255],
    7: [200, 40, 255],
    8: [90, 30, 150],
    9: [255, 0, 255],
    10: [255, 150, 255],
    11: [75, 0, 75],
    12: [75, 0, 175],
    13: [0, 200, 255],
    14: [50, 120, 255],
    15: [0, 175, 0],
    16: [0, 60, 135],
    17: [80, 240, 150],
    18: [150, 240, 255],
    19: [0, 0, 255]
}
label_map = {
    1: 'car',
    2: 'bicycle',
    3: 'motorcycle',
    4: 'truck',
    5: 'other-veh',
    6: 'person',
    7: 'bicyclist',
    8: 'motorcyclist',
    9: 'road',
    10: 'parking',
    11: 'sidewalk',
    12: 'other-ground',
    13: 'building',
    14: 'fence',
    15: 'vegetation',
    16: 'trunk',
    17: 'terrain',
    18: 'pole',
    19: 'tr.-sign'
}

# Convert the color map to RGB values between 0 and 1
for k, v in color_map.items():
    color_map[k] = [c / 255 for c in v]

# Create the bar chart with colored bars
fig, ax = plt.subplots()
bars = ax.bar(range(len(data)), data, color=[color_map[i+1] for i in range(len(data))])

# Add labels to the bars
for i, bar in enumerate(bars):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label_map[i+1], ha='center', va='bottom', fontsize=12)

# Set the axis labels and title
ax.set_xlabel('Object Classes', fontsize=18)
ax.set_ylabel('Percentage', fontsize=18)
ax.set_title('Histogram of Object Classes', fontsize=18)
ax.set_xticklabels([
    'car',
    'bicycle',
    'motorcycle',
    'truck',
    'other-veh',
    'person',
    'bicyclist',
    'motorcyclist',
    'road',
    'parking',
    'sidewalk',
    'other-ground',
    'building',
    'fence',
    'vegetation',
    'trunk',
    'terrain',
    'pole',
    'tr.-sign'], rotation=90)

# Show the plot
plt.show()
