"""
Creadt by: Yoan Wallois
Date: 2024-09-28
Purpose: Simple script to convert a database of climate extreme events to climate forecast events.
"""
#%% Imports
import pandas as pd
import numpy as np 
from pathlib import Path

# %% Import databases on extreme climate events


with open(Path().joinpath('global_climate_events_economic_impact_2020_2025.csv'), 'r', encoding='utf-8') as f:
    ExtremeEvents = pd.read_csv(f, sep=',')

EMDAT = pd.read_excel(Path().joinpath("public_emdat_custom_request_2025-09-28_8317efb3-ee1c-4a60-8ccd-023043de642f.xlsx"), engine='openpyxl')
EMDAT['Location'] = EMDAT['Location'].apply(lambda d: str(d).split(', '))


#%% Add missing location coordonates for extrem event

from ollama import chat
from ollama import ChatResponse

def find_coordonates(location_area):
  response: ChatResponse = chat(model='gpt-occ:20b-cloud', messages=[{
    'role': 'user',
    'content':  'Give me precisely just the latitude and the longitude of' + location_area + "?",
  },])
  return response.message.content


EMDAT['Location_coordonates'] = EMDAT['Location'].apply(lambda location_list: [find_coordonates(location_i) for location_i in location_list])

# %% 