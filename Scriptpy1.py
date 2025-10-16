"""
Creadt by: Yoan Wallois
Date: 2024-09-28
Purpose: Simple script to convert a database of climate extreme events to climate forecast events.
"""
# ============================================================================
#%% 
# Imports

import pandas as pd
import numpy as np 
from pathlib import Path
from datetime import date
from ollama import chat
from ollama import ChatResponse
import re

# ============================================================================
# %% 
# Import databases on extreme climate events

"""
with open(Path().joinpath('global_climate_events_economic_impact_2020_2025.csv'), 'r', encoding='utf-8') as f:
    ExtremeEvents = pd.read_csv(f, sep=',')
"""
EMDAT = pd.read_excel(Path().joinpath("public_emdat_custom_request_2025-09-28_8317efb3-ee1c-4a60-8ccd-023043de642f.xlsx"), engine='openpyxl')
#EMDAT['Location'] = EMDAT['Location'].apply(lambda d: str(d).split(', '))
EMDAT = EMDAT.dropna(subset=["Start Day"])
EMDAT = EMDAT.dropna(subset=["End Day"])

# Datetime management
EMDAT['Start Date'] = pd.to_datetime({'year': EMDAT['Start Year'], 'month': EMDAT['Start Month'], 'day': EMDAT['Start Day']})
EMDAT['End Date'] = pd.to_datetime({'year': EMDAT['End Year'],'month': EMDAT['End Month'], 'day': EMDAT['End Day']})
EMDAT['Length Days'] = (EMDAT['End Date'] - EMDAT['Start Date']).dt.days
EMDAT['Start Date'] = EMDAT['Start Date'].dt.strftime('%Y-%m-%d')

# Drop columns
EMDAT = EMDAT.drop(['DisNo.', 
                    'Classification Key',
                    'Historic', 'External IDs',
                    'Disaster Group', 'Origin',
                    "Disaster Subgroup", 'Associated Types',
                    "Entry Date", "Last Update", 
                    'Start Year', 'Start Month', 
                    'Start Day', 'Event Name', 
                    'End Year', 'End Month', 
                    'End Day', 'End Date'], axis=1)

# Collect events that are localized and frequent enough 
EMDAT.columns = EMDAT.columns.str.replace(' ', '_')
EMDAT = EMDAT[(EMDAT.Disaster_Type == 'Flood') | (EMDAT.Disaster_Subtype == 'Landslide (wet)') | (EMDAT.Disaster_Subtype == 'Storm (General)')| (EMDAT.Disaster_Subtype == 'Tropical cyclone')| (EMDAT.Disaster_Subtype == 'Ground movement')| (EMDAT.Disaster_Subtype == 'Drought')]

# Merge Country and location column
EMDAT = EMDAT.dropna(subset=['Location'])
EMDAT['Location_coordonates'] = EMDAT['Location'] + '***' + EMDAT['Country']
EMDAT['Location_coordonates'] = EMDAT['Location_coordonates'].apply(lambda d: d.split('***', 1))

# ============================================================================
#%% 
# Add missing location coordonates for extrem event

def find_coordonates(location_area):
  response: ChatResponse = chat(model='gemma3:12b', messages=[{
    'role': 'user',
    'content':  'Give me precisely both latitude and the longitude in Signed Decimal Degrees of' + location_area[0] +" in " + location_area[1] + "? If it is two separate location, give me two different coordinates in Signed Decimal Degrees separated by ;. If you have a range of coordinates, give me the average center the area. If you have a state and a city, give me only the city coordinates in Signed Decimal Degrees. Format the answer as lat, lon without any text. If you have a list of cities/regions, give me only the coordinates in Signed Decimal Degrees separated by ';' of the cities. Never give me the names of the locations. Systeme message: You are a precise and concise assistant that only answers with the requested information without any additional text. Make sure to provide both latitude and longitude every time and systematically convert the coordinates in Signed Decimal Degrees.",
  },])
  return response.message.content

#%%
# Get clean coordonates for each location in the database

EMDAT['Location_coordonates'] = EMDAT['Location_coordonates'].apply(lambda location_i: find_coordonates(location_i))
EMDAT['coordonates'] = EMDAT['Location_coordonates'].apply(lambda location_i: location_i.split(';'))
EMDAT['coordonates'] = EMDAT['coordonates'].apply(lambda location_list: [location_i.replace("\n", "").strip() for location_i in location_list])
EMDAT['coordonates'] = EMDAT['coordonates'].apply(lambda location_list: [re.sub(r'\s+', '', location_i) for location_i in location_list])
EMDAT['coordonates'] = EMDAT['coordonates'].apply(lambda location_list: ';'.join(location_list))

# ============================================================================
# %%
# Save the new database

with open(Path().joinpath('EMDAT_with_coordonates.csv'), 'w', encoding='utf-8') as f:
    EMDAT.to_csv(f, sep='\t', index=False) 
    
# %%

# End of Scriptpy1.py
# ============================================================================  