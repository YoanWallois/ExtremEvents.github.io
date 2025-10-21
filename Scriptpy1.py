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

from ollama import Client

client = Client()
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
  response: ChatResponse = chat(model='gpt-oss:120b-cloud', messages=[{
    'role': 'user',
    'content':  "First I want precisely both latitude and the longitude of " + location_area[0] + " in " + location_area[1] + "? If it is separate locations, give me different coordinates in Signed Decimal Degrees separated by ;. If you have more than 3 city coordinates, give me the average center the area. If you have a state name, give me the coordinates of its center. If you have a state/district and a city that have approximatively the same name, give me only the city coordinates unless it clear that the state/district is mentioned. Never give me the names of the locations. Secondly, I want you to make sure you convert it to Signed Decimal Degrees. Then, I want you to format it as such: 'latitude,longitude;latitude,longitude', with r'^-?\d+(.\d+)?,-?\d+(.\d+)?$' format for latitude and longitude. Only answers with the coordinates without any additional text. Be careful to respect the format.",
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
# Testing and re-running the function for missing/invalid coordonates

# Import sample data from csv file
with open('EMDAT_with_coordonates.csv', 'r', encoding='utf-8') as file:
    EMDAT = pd.read_csv(file, sep='\t')


def find_coordonates_reboot(location_area):
    messages = [
        {
    'role': 'user',
    'content': "First I want precisely both latitude and the longitude of " + location_area[0] + " in " + location_area[1] + "? If it is separate locations, give me different coordinates in Signed Decimal Degrees separated by ;. If you have more than 3 city coordinates, give me the average center the area. If you have a state name, give me the coordinates of its center. If you have a state/district and a city that have approximatively the same name, give me only the city coordinates unless it clear that the state/district is mentioned. Never give me the names of the locations. Secondly, I want you to make sure you convert it to Signed Decimal Degrees. Then, I want you to format it as such: 'latitude,longitude;latitude,longitude', with r'^-?\d+(.\d+)?,-?\d+(.\d+)?$' format for latitude and longitude. Only answers with the coordinates without any additional text. Be careful to respect the format.",
    },
    ]
    return "".join([part['message']['content'] for part in client.chat('gpt-oss:120b-cloud', messages=messages, stream=True)])


#%%
# Function to process invalid coordinates

def process_invalid_coordinates(df):
    """
    Process and clean coordinates in a dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'coordonates', 'coordonates1', 'Location_coordonates', 
        'Location', and 'Country' columns
    find_coordonates_reboot : callable
        Function that finds coordinates from a coordinate string
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with processed and validated coordinates
    """
    
    # Step 1: Verify that we have both latitude and longitude and correct missing formats
    df['coordonates_validity'] = df['coordonates'].apply(lambda x: (
      all(re.match(r'^-?\d+(\.\d+)?,-?\d+(\.\d+)?$', part) is not None for part in x.split(';') if len(part) > 0) if pd.notna(x) else False
    ))

    print((df['coordonates_validity'] != False).value_counts()) # Counterintuitive but ok
    
    df['coordonates'] = df.apply(
        lambda row: [row['Location'], row['Country']] 
            if row['coordonates_validity'] == False 
            else row['coordonates'],
        axis=1
    )
    
    df['Location_coordonates1'] = df.apply(
        lambda row: find_coordonates_reboot(row['coordonates']) 
            if row['coordonates_validity'] == False 
            else row['Location_coordonates'],
        axis=1
    )
    
    
    # Step 2: Re-fetch coordinates for invalid entries and clean them
    df['Location_coordonates1'] = df.apply(
        lambda row: find_coordonates_reboot(row['coordonates']) 
            if row['coordonates_validity'] == False 
            else row['Location_coordonates'],
        axis=1
    )
    
    df['coordonates'] = df['Location_coordonates1'].apply(
        lambda location_i: location_i.split(';')
    )
    
    df['coordonates'] = df['coordonates'].apply(
        lambda location_list: [location_i.replace("\n", "").strip() 
                              for location_i in location_list]
    )
    
    df['coordonates'] = df['coordonates'].apply(
        lambda location_list: [re.sub(r'\s+', '', location_i) 
                              for location_i in location_list]
    )
    
    df['coordonates'] = df['coordonates'].apply(
        lambda location_list: ';'.join(location_list)
    )
    
    return df
    

#%%
# Loop to process all invalid coordinates until all are valid

EMDAT = process_invalid_coordinates(EMDAT)

#%% 

EMDAT['coordonates_validity'] = EMDAT['coordonates'].apply(lambda x: (
      all(re.match(r'^-?\d+(\.\d+)?,-?\d+(\.\d+)?$', part) is not None for part in x.split(';') if len(part) > 0) if pd.notna(x) else False
    ))

# %% 
# Save the new database with corrected coordinates
with open(Path().joinpath('EMDAT_with_coordonates.csv'), 'w', encoding='utf-8') as f:
    EMDAT.to_csv(f, sep='\t', index=False) 


# End of Scriptpy1.py
# ============================================================================  
# %%
