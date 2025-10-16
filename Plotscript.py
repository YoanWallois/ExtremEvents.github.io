# %%
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


# Verification that we have both latitude and longitude and correct missing formats
df['coordonates1'] = (df['coordonates'].apply(lambda x: all(',' in part for part in x.split(';') if len(part) > 0) if pd.notna(x) else False)|
                      df['coordonates'].str.contains('[a-zA-Z]', na=False, regex=False)
                      )

df['coordonates'] = df.apply(
    lambda row: [row['Location'], row['Country']] if row['coordonates1'] == False else row['coordonates'],
    axis=1
)

# %%
# Re-run the function to get coordinates using Ollama API

def find_coordonates(location_area):
  response: ChatResponse = chat(model='gemma3:12b', messages=[{
    'role': 'user',
    'content':  'Make sure to really follow my instruction, you made a mistake last time. Give me precisely both latitude and the longitude in Signed Decimal Degrees of' + location_area[0] +" in " + location_area[1] + "? If it is two separate location, give me two different coordinates in Signed Decimal Degrees separated by ;. If you have more that 15 of coordinates, give me the average center the area. If you have a state and a city, give me only the city coordinates in Signed Decimal Degrees. Format the answer as lat, lon without any text. If you have a list of cities/regions, give me only the coordinates in Signed Decimal Degrees separated by ';' of the cities. Never give me the names of the locations. Systeme message: You are a precise and concise assistant that only answers with the requested information without any additional text. Make sure to provide both latitude and longitude every time and systematically convert the coordinates in Signed Decimal Degrees.",
  },])
  return response.message.content


df['Location_coordonates'] = df.apply(
    lambda row: find_coordonates(row['coordonates']) if row['coordonates1'] == False else row['Location_coordonates'],
    axis=1
)


#%%

df['Location_coordonates'] = df['Location_coordonates'].apply(lambda location_i: find_coordonates(location_i))
df['coordonates'] = df['Location_coordonates'].apply(lambda location_i: location_i.split(';'))
df['coordonates'] = df['coordonates'].apply(lambda location_list: [location_i.replace("\n", "").strip() for location_i in location_list])
df['coordonates'] = df['coordonates'].apply(lambda location_list: [re.sub(r'\s+', '', location_i) for location_i in location_list])
df['coordonates'] = df['coordonates'].apply(lambda location_list: ';'.join(location_list))


#%%

df['coordonates'] = df['coordonates'].apply(lambda location_i: location_i.split(';')[0])
df[['latitude', 'longitude']] = df['coordonates'].str.split(',', expand=True)

df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

# ============================================================================
# PLOTTING METHOD 1: Using Cartopy (Recommended)
# ============================================================================
# Cartopy provides professional map projections and geographic features
# Install with: pip install cartopy

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Create figure with high resolution
fig = plt.figure(figsize=(16, 10), dpi=200)

# Set up map projection (PlateCarree = standard lat/lon grid)
ax = plt.axes(projection=ccrs.PlateCarree())

# ---- Add map background features ----
# Land areas in light color
ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='none')

# Ocean areas in soft blue
ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff', edgecolor='none')

# Coastlines with subtle dark line
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#666666', alpha=0.7)

# Country borders (lighter and transparent)
ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='#999999', alpha=0.4)

# ---- Add gridlines ----
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                    alpha=0.3, linestyle='--')
gl.top_labels = False    # Don't show labels on top
gl.right_labels = False  # Don't show labels on right

# ---- Plot the coordinate points ----
scatter = ax.scatter(
    df['longitude'], 
    df['latitude'],
    color='#d62828',           # Deep red color
    s=50,                      # Point size
    alpha=0.8,                  # Slight transparency
    edgecolors='white',         # White border around points
    linewidths=1.5,             # Border width
    transform=ccrs.PlateCarree(),  # Match coordinate system
    zorder=5                    # Draw on top of map features
)

# ---- Configure map extent ----
ax.set_global()  # Show entire world

# ---- Add title and styling ----
plt.title(
    'Geographic Coordinates of extreme climate events on World Map', 
    fontsize=18, 
    fontweight='bold',
    pad=20,
    color='#333333'
)

# Add count of points as subtitle
ax.text(
    0.5, 0.95, 
    f'Total locations: {len(df)}',
    transform=fig.transFigure,
    ha='center',
    fontsize=11,
    color='#666666',
    style='italic'
)

# Adjust layout to minimize white space
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)
plt.show()

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

df = pd.DataFrame({
    'latitude': [48.8566, 40.7128, -33.8688, 35.6762, 51.5074, -23.5505, 55.7558, 19.4326],
    'longitude': [2.3522, -74.0060, 151.2093, 139.6503, -0.1278, -46.6333, 37.6173, -99.1332],
    'year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'location_name': ['Paris', 'New York', 'Sydney', 'Tokyo', 'London', 'São Paulo', 'Moscow', 'Mexico City']
})

# Sort dataframe by year to ensure proper animation sequence
df = df.sort_values('year').reset_index(drop=True)

# Get year range for animation
min_year = int(df['year'].min())
max_year = int(df['year'].max())
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

try:
    # Save as GIF (requires pillow: pip install pillow)
    anim.save('Figures/location_timeline.gif', writer='pillow', fps=1, dpi=200)
    print("✓ Animation saved as 'location_timeline.gif'")
except Exception as e:
    print(f"✗ Error saving GIF: {e}")
    print("  Make sure pillow is installed: pip install pillow")

# ========================================================================
# OPTIONAL: Save animation as video or GIF
# ========================================================================
# The GIF is now automatically saved above. 
# Uncomment the line below to also save as MP4

# Save as MP4 (requires ffmpeg: conda install ffmpeg or apt-get install ffmpeg)
# anim.save('location_timeline.mp4', writer='ffmpeg', fps=1, dpi=150)


# %%
