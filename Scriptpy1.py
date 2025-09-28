"""
Creadt by: Yoan Wallois
Date: 2024-09-28
Purpose: Simple script to convert a database of climate extreme events to climate forecast events.
"""
#%% Imports
import pandas as pd
import numpy as np 
from pathlib import Path

# %%


with open(Path().joinpath('global_climate_events_economic_impact_2020_2025.csv'), 'r', encoding='utf-8') as f:
    ExtremeEvents = pd.read_csv(f, sep=',')

EMDAT = pd.read_excel(Path().joinpath("public_emdat_custom_request_2025-09-28_8317efb3-ee1c-4a60-8ccd-023043de642f.xlsx"), engine='openpyxl')
