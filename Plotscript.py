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
    
df = df[df['coordonates_validity'] != False]
df['coordonates'] = df['coordonates'].apply(lambda location_i: location_i.split(';')[0])

# Clean and split the coordinates
#df['coordonates'] = df['coordonates'].apply(lambda location_i: location_i.split(';')[0])
df[['latitude', 'longitude']] = df['coordonates'].str.split(',', expand=True)

df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

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
    fig = plt.figure(figsize=(16, 10), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff', edgecolor='none')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#666666', alpha=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='#999999', alpha=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                        alpha=0.3, linestyle='--')
    gl.top_labels = False    # Don't show labels on top
    gl.right_labels = False  # Don't show labels on right
    scatter = ax.scatter(
        df['longitude'], 
        df['latitude'],
        color= color,           # Deep red color
        s=30,                      # Point size
        alpha=0.8,                  # Slight transparency
        edgecolors='white',         # White border around points
        linewidths=1,             # Border width
        transform=ccrs.PlateCarree(),  # Match coordinate system
        zorder=5                    # Draw on top of map features
    )
    ax.set_global() 
    plt.title(
        'Geographic Coordinates of ' + df['Disaster_Subtype'].unique()[0] + ' on World Map', 
        fontsize=18, 
        fontweight='bold',
        pad=20,
        color='#333333'
    )
    ax.text(
        0.5, 0.95, 
        f'Total locations: {len(df)}',
        transform=fig.transFigure,
        ha='center',
        fontsize=11,
        color='#666666',
        style='italic'
    )
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)
    plt.savefig('Figures/world_map_coordinates_' + df['Disaster_Subtype'].unique()[0][0:4] + '.png', dpi=300, bbox_inches='tight')

plot_world_map(df[df['Disaster_Subtype'] == 'Flood (General)'], "#200ec2")
plot_world_map(df[df['Disaster_Subtype'] == 'Riverine flood'], "#7a0ec2")
plot_world_map(df[df['Disaster_Subtype'] == 'Tropical cyclone'], "#cc11b3")
plot_world_map(df[df['Disaster_Subtype'] == 'Flash flood'], "#0e62c2")
plot_world_map(df[df['Disaster_Subtype'] == 'Ground movement'], "#c2500e")
plot_world_map(df[df['Disaster_Subtype'] == 'Landslide (wet)'], "#38c20e")
plot_world_map(df[df['Disaster_Subtype'] == 'Storm (General)'], "#c20e0e")
plot_world_map(df[df['Disaster_Subtype'] == 'Coastal flood'], "#077377")
plot_world_map(df[df['Disaster_Subtype'] == 'Drought'], "#00000000")

# ============================================================================
# OPTIONAL: Save the figure to file
# ============================================================================
# Uncomment the line below to save the plot as a high-resolution image
# plt.savefig('world_map_coordinates.png', dpi=300, bbox_inches='tight')

# %%

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Configure matplotlib for interactive display
plt.ion()  # Turn on interactive mode

# ============================================================================
# SAMPLE DATA - Replace with your actual dataframe
# ============================================================================
# Your dataframe should have: 'latitude', 'longitude', 'year', 'location_name'

# Extract month year from datetime column
df['date'] = pd.to_datetime(df['Start_Date'])

#%%

# Sort dataframe by year to ensure proper animation sequence
df['year'] = df['date'].dt.strftime('%Y-%m')

# Get year range for animation
min_year = df['year'].min()
max_year = df['year'].max()
years = range(min_year, max_year + 1)

# ============================================================================
# ANIMATION SETUP
# ============================================================================

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Create figure with high resolution
fig = plt.figure(figsize=(16, 10), dpi=200)

# Set up map projection
ax = plt.axes(projection=ccrs.PlateCarree())

# ---- Add map background features ----
ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='none')
ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff', edgecolor='none')
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#666666', alpha=0.7)
ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='#999999', alpha=0.4)

# ---- Add gridlines ----
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                    alpha=0.3, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Set global extent
ax.set_global()

# ---- Initialize empty plot elements ----
scatter = ax.scatter([], [], color='#d62828', s=150, alpha=0.8,
                    edgecolors='white', linewidths=2,
                    transform=ccrs.PlateCarree(), zorder=5)

# Text annotations list (will hold location labels)
text_annotations = []

# Year display (large text showing current year)
year_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                    fontsize=32, fontweight='bold', color='#333333',
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='white', alpha=0.8, edgecolor='#cccccc'))

# Title
title = plt.title('Geographic Locations Timeline', fontsize=18, 
                    fontweight='bold', pad=20, color='#333333')

# ============================================================================
# ANIMATION FUNCTION
# ============================================================================
def animate(frame_year):
    """
    Update function called for each frame of the animation
    frame_year: the current year being displayed
    """
    # Clear previous text annotations
    for txt in text_annotations:
        txt.remove()
    text_annotations.clear()
    
    # Filter data for current year only
    current_data = df[df['year'] == frame_year]
    
    # Update scatter plot with current year's points
    if len(current_data) > 0:
        scatter.set_offsets(current_data[['longitude', 'latitude']].values)
        
        # Add text labels for each location
        for idx, row in current_data.iterrows():
            # Create text annotation with location name
            txt = ax.text(
                row['longitude'], 
                row['latitude'] + 3,  # Offset text above the point
                row['location_name'],
                transform=ccrs.PlateCarree(),
                fontsize=10,
                fontweight='bold',
                color='#d62828',
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='white', 
                            alpha=0.9,
                            edgecolor='#d62828',
                            linewidth=1.5),
                zorder=6
            )
            text_annotations.append(txt)
    else:
        # No data for this year - show empty plot
        scatter.set_offsets([[],[]])
    
    # Update year display
    year_text.set_text(f'{frame_year}')
    
    return [scatter, year_text] + text_annotations

# ============================================================================
# CREATE AND DISPLAY ANIMATION
# ============================================================================
print(f"Creating animation from {min_year} to {max_year}...")

# Create animation
# interval: milliseconds between frames (1000 = 1 second per year)
# repeat: animation will loop continuously
anim = animation.FuncAnimation(
    fig, 
    animate, 
    frames=years,
    interval=1000,  # 1 second per year
    blit=False,  # Changed to False for better compatibility
    repeat=True
)

# Adjust layout to minimize white space
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)

# Display the animation
plt.show(block=False)

# ========================================================================
# SAVE ANIMATION AS GIF
# ========================================================================
print("Saving animation as GIF... This may take a moment.")


# Save as GIF (requires pillow: pip install pillow)
anim.save('Figures/location_timeline.gif', writer='pillow', fps=1, dpi=200)


# ========================================================================
# OPTIONAL: Save animation as video or GIF
# ========================================================================
# The GIF is now automatically saved above. 
# Uncomment the line below to also save as MP4

# Save as MP4 (requires ffmpeg: conda install ffmpeg or apt-get install ffmpeg)
# anim.save('location_timeline.mp4', writer='ffmpeg', fps=1, dpi=150)


# %%
