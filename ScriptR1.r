

# ============================================================================
# SETUP AND CONFIGURATION
# ============================================================================

# Set CRAN repository
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# ============================================================================
# LOAD REQUIRED LIBRARIES
# ============================================================================

# Load required libraries
library(sf)
library(terra)
library(tmap)
library(geodata)
library(blackmarbler)
library(raster)
library(exactextractr)
library(ggplot2)
library(lubridate)
library(httr)
library(jsonlite)
library(rhdf5)
library(terra)
library(lintr)
line_length_linter(length = 80L)

# ============================================================================
# API CREDENTIALS
# ============================================================================
# Your NASA Earthdata credentials
DAYLY_UPDATED_API_KEY <- ""

# ============================================================================
# VIIRS NIGHTLIGHT DATA DOWNLOAD
# ============================================================================


