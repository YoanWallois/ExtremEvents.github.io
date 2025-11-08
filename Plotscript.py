# %%
# Import necessary libraries and data
import pandas as pd
import matplotlib.pyplot as plt
from ollama import chat
from ollama import ChatResponse
import re

# ============================================================================
# SAMPLE DATA - Replace with your actual dataframe
# ============================================================================
# Your dataframe should have 'latitude' and 'longitude' columns
# Latitude: -90 to 90 (negative = South, positive = North)
# Longitude: -180 to 180 (negative = West, positive = East)

# Import sample data from csv file
with open('EMDAT_with_coordonates.csv', 'r', encoding='utf-8') as file:
    df = pd.read_csv(file, sep='\t')
    
df['coordonates_validity'] = df['coordonates'].apply(lambda x: (
      all(re.match(r'^-?\d+(\.\d+)?,-?\d+(\.\d+)?$', part) is not None for part in x.split(';') if len(part) > 0) if pd.notna(x) else False
    ))

df = df[df['coordonates_validity'] != False]
df['coordonates'] = df['coordonates'].apply(lambda location_i: location_i.split(';')[0])

# Clean and split the coordinates
#df['coordonates'] = df['coordonates'].apply(lambda location_i: location_i.split(';')[0])
df[['latitude', 'longitude']] = df['coordonates'].str.split(',', expand=True)

df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

df['Disaster_Type'] = df['Disaster_Type'].apply(lambda x: re.sub(r'Mass movement \(wet\)', 'Landslide', x))

df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce')
df['Mdate'] = df['Start_Date'].dt.strftime('%Y-%m')

# Create a clean location name for labeling
df['location_name'] = df['Location'].apply(lambda d: re.sub(r'\([^)]*\)', '', d).strip())
df['location_name'] = df['location_name'].apply(lambda d: str(d).split(';')[0])
df['location_name'] = df['location_name'].apply(lambda d: str(d).split(',')[0])

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%
# ============================================================================
# PLOTTING METHOD 1: Using Cartopy (Recommended)
# ============================================================================
# Cartopy provides professional map projections and geographic features
# Install with: pip install cartopy

import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_world_map(df, color):
    """
    Plots geographic coordinates on a world map using Cartopy.
    df: DataFrame with 'latitude' and 'longitude' columns
    """
    # Create  figure with high resolution
    fig = plt.figure(figsize=(19, 9), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff', edgecolor='none')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#666666', alpha=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='#999999', alpha=0.4)
    gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', 
                        alpha=0.3, linestyle='--')
    scatter = ax.scatter(
        df['longitude'], 
        df['latitude'],
        color= color,           # Deep red color
        s=80,                      # Point size
        alpha=0.8,                  # Slight transparency
        edgecolors='white',         # White border around points
        linewidths=1.5,             # Border width
        transform=ccrs.PlateCarree(),  # Match coordinate system
        zorder=5                    # Draw on top of map features
    )
    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree()) 
    plt.title(
        'Geographic Coordinates of ' + df['Disaster_Subtype'].unique()[0] + ' on World Map', 
        fontsize=18, 
        fontweight='bold',
        pad=20,
        color='#333333',
        fontfamily='Times New Roman'
    )
    ax.text(
        0.5, 0.05, 
        f'Total locations: {len(df)}',
        transform=fig.transFigure,
        ha='center',
        fontsize=11,
        color='#666666',
        style='italic'
    )
    plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.02)
    plt.savefig('Figures/world_map_coordinates_' + df['Disaster_Subtype'].unique()[0][0:4] + '.png', dpi=300, bbox_inches='tight')

plot_world_map(df[df['Disaster_Subtype'] == 'Flood (General)'], "#200ec2")
plot_world_map(df[df['Disaster_Subtype'] == 'Riverine flood'], "#7a0ec2")
plot_world_map(df[df['Disaster_Subtype'] == 'Tropical cyclone'], "#cc11b3")
plot_world_map(df[df['Disaster_Subtype'] == 'Flash flood'], "#0e62c2")
plot_world_map(df[df['Disaster_Subtype'] == 'Ground movement'], "#c2500e")
plot_world_map(df[df['Disaster_Subtype'] == 'Landslide (wet)'], "#38c20e")
plot_world_map(df[df['Disaster_Subtype'] == 'Storm (General)'], "#c20e0e")
plot_world_map(df[df['Disaster_Subtype'] == 'Coastal flood'], "#077377")


#%%

# Configure matplotlib for interactive display
plt.ion()  # Turn on interactive mode
# ============================================================================
# Define a distinct color for each Disaster_Type
disasterType_color = {
    'Earthquake': '#e63946',        # Red
    'Storm': '#f77f00',          # Orange
    'Flood': '#118ab2', # Blue
    'Landslide': '#9d4edd', # Purple
}

# Sort dataframe by date to ensure proper animation sequence
df = df.sort_values('Mdate').reset_index(drop=True)

# Get unique dates for animation (in chronological order)
unique_Mdates = sorted(df['Mdate'].unique())

print(f"Animation will show {len(unique_Mdates)} time periods from {unique_Mdates[0]} to {unique_Mdates[-1]}")
print(f"\nDisaster_Type color scheme:")
for Disaster_Type, color in disasterType_color.items():
    if Disaster_Type in df['Disaster_Type'].values:
        print(f"  {Disaster_Type}: {color}")

# ============================================================================
# ANIMATION SETUP
# ============================================================================
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Create figure with high resolution
fig = plt.figure(figsize=(19, 9), dpi=200)

# Set up map projection
ax = plt.axes(projection=ccrs.PlateCarree())

# ---- Add map background features ----
ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='none')
ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff', edgecolor='none')
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#666666', alpha=0.7)
ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='#999999', alpha=0.4)

# ---- Add gridlines ----
gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', 
                    alpha=0.3, linestyle='--')

# Set extent to exclude Antarctica
ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())

# ---- Initialize empty plot elements ----
scatter = ax.scatter([], [], s=80, alpha=0.8,
                    edgecolors='white', linewidths=1.5,
                    transform=ccrs.PlateCarree(), zorder=5)

# Text annotations list (will hold location labels)
text_annotations = []

# Legend for Disaster_Types
legend_elements = []
for Disaster_Type, color in disasterType_color.items():
    if Disaster_Type in df['Disaster_Type'].values:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=13,
                                            label=Disaster_Type, markeredgecolor='white',
                                            markeredgewidth=1.2))

# Add legend to the plot
legend = ax.legend(handles=legend_elements, loc='lower left',
                    framealpha=0.9, title='Disaster types',
                    title_fontsize=20, prop={'family': 'Times New Roman', 'size': 16})
legend.get_title().set_fontfamily('Times New Roman')

# Year display (large text showing current Mdate)
Mdate_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                    fontsize=28, fontweight='bold', color='#333333',
                    verticalalignment='top', fontfamily='Times New Roman',
                    bbox=dict(boxstyle='round',
                    facecolor='white', alpha=0.8, edgecolor='#cccccc'))

# Title
title = plt.title('Geographic Locations Timeline', fontsize=18, 
                    fontweight='bold', pad=20, color='#333333',
                    fontfamily='Times New Roman')

# ============================================================================
# ANIMATION FUNCTION
# ============================================================================
def animate(frame_idx):
    """
    Update function called for each frame of the animation
    frame_idx: index of the current date in unique_dates list
    """
    current_Mdate = unique_Mdates[frame_idx]
    
    # Clear previous text annotations
    for txt in text_annotations:
        txt.remove()
    text_annotations.clear()
    
    # Filter data for current date only
    current_data = df[df['Mdate'] == current_Mdate]
    
    # Update scatter plot with current date's points
    if len(current_data) > 0:
        # Get colors for each point based on Disaster_Type
        colors = [disasterType_color.get(cont, '#d62828') for cont in current_data['Disaster_Type']]
        
        # Update scatter plot positions and colors
        scatter.set_offsets(current_data[['longitude', 'latitude']].values)
        scatter.set_color(colors)
        
        # Track occupied regions to avoid overlaps
        occupied_regions = []
        
        # Function to check if a label position overlaps with existing labels
        def check_overlap(new_lon, new_lat, occupied_list, threshold_lon=5.5, threshold_lat=3.5):
            """
            Check if new label position overlaps with any occupied region.
            Uses elliptical distance metric for more natural collision detection.
            """
            for occ_lon, occ_lat in occupied_list:
                # Calculate weighted distance (accounts for different thresholds)
                lon_dist = abs(new_lon - occ_lon) / threshold_lon
                lat_dist = abs(new_lat - occ_lat) / threshold_lat
                
                # Use elliptical distance: if sum of normalized distances < 1, labels overlap
                elliptical_distance = (lon_dist ** 2 + lat_dist ** 2) ** 0.5
                
                if elliptical_distance < 1.0:  # Overlap detected
                    return True
            return False
        
        # Define offset directions (radiating outward from point) - more positions for better spacing
        # Format: (lon_offset, lat_offset)
        offset_directions = [
            (0, 2.5),      # Above
            (3, 1.5),      # Above-right
            (3, -1.5),     # Below-right
            (0, -2.5),     # Below
            (-3, -1.5),    # Below-left
            (-3, 1.5),     # Above-left
            (5, 2.5),      # Far above-right
            (5, -2.5),     # Far below-right
            (-5, 2.5),     # Far above-left
            (-5, -2.5),    # Far below-left
            (0, 4.5),      # Far above
            (0, -4.5),     # Far below
            (6, 0),        # Far right
            (-6, 0),       # Far left
            (7, 1),        # Very far above-right
            (-7, 1),       # Very far above-left
            (7, -1),       # Very far below-right
            (-7, -1),      # Very far below-left
            (2, 4),        # Mid-right above
            (-2, 4),       # Mid-left above
            (2, -4),       # Mid-right below
            (-2, -4),      # Mid-left below
            (8, 0),        # Extremely far right
            (-8, 0),       # Extremely far left
            (4, 3),        # Mid-far above-right
            (-4, 3),       # Mid-far above-left
            (4, -3),       # Mid-far below-right
            (-4, -3),      # Mid-far below-left
        ]
        
        # Sort data by latitude (top to bottom) and longitude for better spacing
        sorted_data = current_data.sort_values(['latitude', 'longitude'], ascending=[False, True])
        
        # Add text labels for each location
        for idx, row in sorted_data.iterrows():
            # Get color for this Disaster_Type
            color = disasterType_color.get(row['Disaster_Type'], '#d62828')
            
            # Try each offset direction until finding a non-overlapping position
            best_offset = offset_directions[0]
            found_space = False
            
            for lon_offset, lat_offset in offset_directions:
                label_lon = row['longitude'] + lon_offset
                label_lat = row['latitude'] + lat_offset
                
                # Check if this position overlaps with any occupied region using improved detection
                if not check_overlap(label_lon, label_lat, occupied_regions):
                    best_offset = (lon_offset, lat_offset)
                    found_space = True
                    break
            
            # Use the best offset found (either non-overlapping or closest to original)
            label_lon = row['longitude'] + best_offset[0]
            label_lat = row['latitude'] + best_offset[1]
            
            occupied_regions.append((label_lon, label_lat))
            
            # Create text annotation with location name
            txt = ax.text(
                label_lon, 
                label_lat,
                row['location_name'],
                transform=ccrs.PlateCarree(),
                fontsize=10.5,
                fontfamily='Times New Roman',
                fontweight='bold',
                color=color,
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.35', 
                            facecolor='white', 
                            alpha=0.85,
                            edgecolor=color,
                            linewidth=0.85),
                zorder=6
            )
            text_annotations.append(txt)
    else:
        # No data for this date - show empty plot
        scatter.set_offsets([[],[]])
        scatter.set_color([])
    
    # Update Mdate display
    Mdate_text.set_text(f'{current_Mdate}')
    
    return [scatter, Mdate_text] + text_annotations

# ============================================================================
# CREATE AND DISPLAY ANIMATION
# ============================================================================
print(f"Creating animation for {len(unique_Mdates)} time periods...")

# Create animation
# interval: milliseconds between frames (1000 = 1 second per month)
# frames: number of frames (one for each unique date)
# repeat: animation will loop continuously
anim = animation.FuncAnimation(
    fig, 
    animate, 
    frames=len(unique_Mdates),
    interval=1000,  # 1 second per date
    blit=False,  # Changed to False for better compatibility
    repeat=True
)

# Adjust layout to minimize white space
plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.02)


# Display the animation
plt.show(block=False)

# ========================================================================
# SAVE ANIMATION AS GIF
# ========================================================================
anim.save("Figures/DynamicWorldEE.gif", writer='pillow', fps=1, dpi=200)


    
# %%
