import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
import os
import h5py
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc
import glob
import math
from dotenv import load_dotenv
import traceback
import sys
import re
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load environment variables
load_dotenv()

# Define color constants
REF_COLOR = "#FFFFD0"  # Light yellow/cream for REF rows
ACTUAL_COLOR = "#C0F0C0"  # Light green for ACTUAL rows
REF_GRAPH_COLOR = "#0080FF"  # Blue for REF graph lines
ACTUAL_GRAPH_COLOR = "#00A080"  # Teal green for ACTUAL graph lines
BORDER_COLOR = "#DDDDDD"  # Light gray for table borders

# --- Setup ---
class Config:
    def __init__(self):
        # Data dimensions - with fallbacks to ensure the app works
        self.data_points = self._get_env_int('DATA_POINTS', 41)
        self.ref_points = self._get_env_int('REF_POINTS', 21)
        
        # Reference data parameters
        self.x_max = self._get_env_float('X_MAX', 100.0)
        self.ref_cycles = self._get_env_float('REF_CYCLES', 2.0)
        self.amplitude = self._get_env_float('AMPLITUDE', 10.0)
        self.offset = self._get_env_float('OFFSET', 50.0)
        self.profile_scale = self._get_env_float('PROFILE_SCALE', 0.5)
        
        # Noise parameters
        self.x_noise_scale = self._get_env_float('X_NOISE_SCALE', 0.5)
        self.z_noise_scale = self._get_env_float('Z_NOISE_SCALE', 1.0)
        
        # App settings
        self.secret_key = os.getenv('SECRET_KEY', 'h5_visualization_dashboard')
        self.h5_files_dir = os.getenv('H5_FILES_DIR', './data')
        
        # Data rows will be set dynamically when data is loaded
        self.data_rows = None
        self.rows_per_page = 10  # Number of rows to display per page
        self.display_points = 3  # Number of points to display in the table

    def _get_env_int(self, name, default=None):
        """Get integer environment variable with fallback"""
        value = os.getenv(name)
        if value is not None:
            return int(value)
        return default
    
    def _get_env_float(self, name, default=None):
        """Get float environment variable with fallback"""
        value = os.getenv(name)
        if value is not None:
            return float(value)
        return default

config = Config()

# Global data store
data_store = {}
# Store the current file name globally
current_file_name = None
# Store file history
file_history = []
# Store error messages for display
error_message = None

# --- Reference points with midpoints ---
def generate_reference_display(ref_x, ref_z):
    """Generate reference display points with midpoints"""
    display_x, display_z, is_midpoint = [], [], []

    for i in range(len(ref_x) - 1):
        display_x.append(ref_x[i])
        display_z.append(ref_z[i])
        is_midpoint.append(False)
        
        # Calculate midpoint
        mx = (ref_x[i] + ref_x[i+1]) / 2
        mz = (ref_z[i] + ref_z[i+1]) / 2
        display_x.append(mx)
        display_z.append(mz)
        is_midpoint.append(True)

    # Add last point
    display_x.append(ref_x[-1])
    display_z.append(ref_z[-1])
    is_midpoint.append(False)

    return display_x, display_z, is_midpoint

# --- Helper functions to find data in HDF5 file ---
def find_datasets_by_pattern(h5_file, pattern):
    """Find datasets in an HDF5 file that match a pattern"""
    results = []
    
    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset) and re.search(pattern, name):
            results.append(name)
    
    h5_file.visititems(visitor_func)
    return results

def find_group_with_pattern(h5_file, pattern):
    """Find groups in an HDF5 file that match a pattern"""
    results = []
    
    def visitor_func(name, node):
        if isinstance(node, h5py.Group) and re.search(pattern, name):
            results.append(name)
    
    h5_file.visititems(visitor_func)
    return results

def find_attribute_in_group(h5_file, group_name, attr_pattern):
    """Find an attribute in a group that matches a pattern"""
    if group_name in h5_file:
        group = h5_file[group_name]
        for attr_name in group.attrs:
            if re.search(attr_pattern, attr_name, re.IGNORECASE):
                return attr_name
    return None

# --- Load H5 content ---
def load_data_from_h5(file_path):
    """Load all data from H5 file and prepare it for display"""
    global data_store, current_file_name, file_history, error_message

    try:
        print(f"[DEBUG] Loading file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            error_message = f"File does not exist: {file_path}"
            print(f"[ERROR] {error_message}")
            return False
            
        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            error_message = f"File is empty: {file_path}"
            print(f"[ERROR] {error_message}")
            return False
            
        try:
            with h5py.File(file_path, "r") as h5_file:
                print(f"[DEBUG] H5 file opened successfully")
                print(f"[DEBUG] H5 file keys: {list(h5_file.keys())}")
                
                # Find the main data group (usually "process data" but could be different)
                main_data_groups = find_group_with_pattern(h5_file, r'process.*data|data')
                if not main_data_groups:
                    error_message = f"Could not find main data group in file: {file_path}"
                    print(f"[ERROR] {error_message}")
                    return False
                
                main_data_group = main_data_groups[0]  # Use the first matching group
                print(f"[DEBUG] Found main data group: {main_data_group}")
                
                # Find screwdown, bending, and profile groups
                screwdown_groups = find_group_with_pattern(h5_file, r'screwdown|screw.*down')
                bending_groups = find_group_with_pattern(h5_file, r'bending|bend')
                profile_groups = find_group_with_pattern(h5_file, r'profile')
                blank_info_groups = find_group_with_pattern(h5_file, r'blank.*info|info')
                
                if not screwdown_groups or not bending_groups or not profile_groups or not blank_info_groups:
                    error_message = f"Missing required data groups in file: {file_path}"
                    print(f"[ERROR] {error_message}")
                    print(f"[DEBUG] Found groups: screwdown={screwdown_groups}, bending={bending_groups}, profile={profile_groups}, blank_info={blank_info_groups}")
                    return False
                
                screwdown_group = screwdown_groups[0]
                bending_group = bending_groups[0]
                profile_group = profile_groups[0]
                blank_info_group = blank_info_groups[0]
                
                # Find x and z datasets for each group
                # Screwdown data
                screwdown_x_datasets = find_datasets_by_pattern(h5_file, f"{screwdown_group}/.*x")
                screwdown_z_datasets = find_datasets_by_pattern(h5_file, f"{screwdown_group}/.*z")
                
                if not screwdown_x_datasets or not screwdown_z_datasets:
                    error_message = f"Missing screwdown x/z datasets in file: {file_path}"
                    print(f"[ERROR] {error_message}")
                    return False
                
                screw_actual_x = h5_file[screwdown_x_datasets[0]][:]
                screw_actual_z = h5_file[screwdown_z_datasets[0]][:]
                
                # Find reference attributes
                screw_ref_x_attr = find_attribute_in_group(h5_file, screwdown_group, r'.*ref.*x')
                screw_ref_z_attr = find_attribute_in_group(h5_file, screwdown_group, r'.*ref.*z')
                
                if not screw_ref_x_attr or not screw_ref_z_attr:
                    error_message = f"Missing screwdown reference attributes in file: {file_path}"
                    print(f"[ERROR] {error_message}")
                    return False
                
                screw_ref_x = h5_file[screwdown_group].attrs[screw_ref_x_attr]
                screw_ref_z = h5_file[screwdown_group].attrs[screw_ref_z_attr]
                
                # Bending data
                bending_x_datasets = find_datasets_by_pattern(h5_file, f"{bending_group}/.*x")
                bending_z_datasets = find_datasets_by_pattern(h5_file, f"{bending_group}/.*z")
                
                if not bending_x_datasets or not bending_z_datasets:
                    error_message = f"Missing bending x/z datasets in file: {file_path}"
                    print(f"[ERROR] {error_message}")
                    return False
                
                bend_actual_x = h5_file[bending_x_datasets[0]][:]
                bend_actual_z = h5_file[bending_z_datasets[0]][:]
                
                # Find reference attributes
                bend_ref_x_attr = find_attribute_in_group(h5_file, bending_group, r'.*ref.*x')
                bend_ref_z_attr = find_attribute_in_group(h5_file, bending_group, r'.*ref.*z')
                
                if not bend_ref_x_attr or not bend_ref_z_attr:
                    error_message = f"Missing bending reference attributes in file: {file_path}"
                    print(f"[ERROR] {error_message}")
                    return False
                
                bend_ref_x = h5_file[bending_group].attrs[bend_ref_x_attr]
                bend_ref_z = h5_file[bending_group].attrs[bend_ref_z_attr]
                
                # Profile data
                profile_x_datasets = find_datasets_by_pattern(h5_file, f"{profile_group}/.*x")
                profile_z_datasets = find_datasets_by_pattern(h5_file, f"{profile_group}/.*z")
                
                if not profile_x_datasets or not profile_z_datasets:
                    error_message = f"Missing profile x/z datasets in file: {file_path}"
                    print(f"[ERROR] {error_message}")
                    return False
                
                profile_actual_x = h5_file[profile_x_datasets[0]][:]
                profile_actual_z = h5_file[profile_z_datasets[0]][:]
                
                # Find reference attributes
                profile_ref_x_attr = find_attribute_in_group(h5_file, profile_group, r'.*ref.*x')
                profile_ref_z_attr = find_attribute_in_group(h5_file, profile_group, r'.*ref.*z')
                
                if not profile_ref_x_attr or not profile_ref_z_attr:
                    error_message = f"Missing profile reference attributes in file: {file_path}"
                    print(f"[ERROR] {error_message}")
                    return False
                
                profile_ref_x = h5_file[profile_group].attrs[profile_ref_x_attr]
                profile_ref_z = h5_file[profile_group].attrs[profile_ref_z_attr]
                
                # Blank info and boolean data
                blank_info_single_datasets = find_datasets_by_pattern(h5_file, f"{blank_info_group}/.*single")
                blank_info_boolean_datasets = find_datasets_by_pattern(h5_file, f"{blank_info_group}/.*boolean")
                
                if not blank_info_single_datasets:
                    # Try to find any dataset in the blank info group
                    all_datasets = find_datasets_by_pattern(h5_file, f"{blank_info_group}/.*")
                    if all_datasets:
                        blank_info_single_datasets = [all_datasets[0]]
                    else:
                        error_message = f"Missing blank info datasets in file: {file_path}"
                        print(f"[ERROR] {error_message}")
                        return False
                
                blank_infos_single = h5_file[blank_info_single_datasets[0]][:]
                
                # Boolean data is optional
                boolean_data = None
                if blank_info_boolean_datasets:
                    boolean_data = h5_file[blank_info_boolean_datasets[0]][:]
                
                # Use the first row from blank_infos_single as blank info identifiers
                blank_infos_0 = blank_infos_single[0, :]
                
                # Ensure dimensions are set
                config.data_rows = blank_infos_0.shape[0]
                print(f"[DEBUG] Data rows: {config.data_rows}")
                
                # Flatten boolean values and align with blank info if available
                boolean_df = None
                if boolean_data is not None:
                    boolean_flat = boolean_data.flatten()
                    boolean_df = pd.DataFrame({'value': boolean_flat}, index=blank_infos_0.astype(int))
                else:
                    # Create a dummy boolean dataframe
                    boolean_df = pd.DataFrame({'value': np.zeros(len(blank_infos_0))}, index=blank_infos_0.astype(int))
                
                # Set data_points dynamically based on actual data
                config.data_points = screw_actual_x.shape[1]  # Use actual data
                print(f"[DEBUG] Data points set to: {config.data_points}")

                config.ref_points = len(screw_ref_x)  # Use actual data
                print(f"[DEBUG] Ref points set to: {config.ref_points}")

        except Exception as e:
            error_message = f"Failed to read H5 file: {str(e)}"
            print(f"[ERROR] {error_message}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            return False

        # Generate reference display points with midpoints
        screw_disp_x, screw_disp_z, screw_mid = generate_reference_display(screw_ref_x, screw_ref_z)
        bend_disp_x, bend_disp_z, bend_mid = generate_reference_display(bend_ref_x, bend_ref_z)
        profile_disp_x, profile_disp_z, profile_mid = generate_reference_display(profile_ref_x, profile_ref_z)

        # Ensure actual data arrays match the reference display points in length
        target_points = len(screw_disp_x)  # Use the length of reference display points
        
        # Resize arrays to ensure consistent dimensions
        # For actual data, pad with the last value if needed
        if screw_actual_x.shape[1] < target_points:
            pad_width = ((0, 0), (0, target_points - screw_actual_x.shape[1]))
            screw_actual_x = np.pad(screw_actual_x, pad_width, mode='edge')
            screw_actual_z = np.pad(screw_actual_z, pad_width, mode='edge')
            bend_actual_x = np.pad(bend_actual_x, pad_width, mode='edge')
            bend_actual_z = np.pad(bend_actual_z, pad_width, mode='edge')
            profile_actual_x = np.pad(profile_actual_x, pad_width, mode='edge')
            profile_actual_z = np.pad(profile_actual_z, pad_width, mode='edge')
        
        # Resize to ensure consistent row dimensions
        screw_actual_x = np.resize(screw_actual_x, (config.data_rows, target_points))
        screw_actual_z = np.resize(screw_actual_z, (config.data_rows, target_points))
        bend_actual_x = np.resize(bend_actual_x, (config.data_rows, target_points))
        bend_actual_z = np.resize(bend_actual_z, (config.data_rows, target_points))
        profile_actual_x = np.resize(profile_actual_x, (config.data_rows, target_points))
        profile_actual_z = np.resize(profile_actual_z, (config.data_rows, target_points))

        # Store data in global data store
        data_store = {
            "screwdown": {
                "label": "Screwdown", 
                "actual_x": screw_actual_x, 
                "actual_z": screw_actual_z, 
                "ref_x": screw_disp_x, 
                "ref_z": screw_disp_z,
                "is_midpoint": screw_mid
            },
            "bending": {
                "label": "Bending", 
                "actual_x": bend_actual_x, 
                "actual_z": bend_actual_z, 
                "ref_x": bend_disp_x, 
                "ref_z": bend_disp_z,
                "is_midpoint": bend_mid
            },
            "profile": {
                "label": "Profile", 
                "actual_x": profile_actual_x, 
                "actual_z": profile_actual_z, 
                "ref_x": profile_disp_x, 
                "ref_z": profile_disp_z,
                "is_midpoint": profile_mid
            },
            "blank_info": {
                "label": "Blank Info",
                "data": blank_infos_0
            },
            "boolean_info": {
                "label": "Boolean Info",
                "data": boolean_df
            }
        }
        
        # Store the current file name globally
        current_file_name = os.path.basename(file_path)
        
        # Add to file history if it's a new file
        if not file_history or file_history[-1] != current_file_name:
            file_history.append(current_file_name)
            # Keep only the last 10 files in history
            if len(file_history) > 10:
                file_history = file_history[-10:]
        
        # Clear any previous error messages
        error_message = None
        
        print(f"[DEBUG] Data loaded successfully. data_store keys: {data_store.keys()}")
        return True
    except Exception as e:
        error_message = f"Failed to load data: {str(e)}"
        print(f"[ERROR] {error_message}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return False

# --- Find available H5 files ---
def find_h5_files():
    """Find all available H5 files in the data directory and other common locations"""
    h5_files = []
    
    # Check in the configured data directory
    h5_files.extend(glob.glob(os.path.join(config.h5_files_dir, '*.h5')))
    
    # Check in other common locations
    common_locations = [
        './',  # Current directory
        '../',  # Parent directory
        './valid/',  # Valid subdirectory
        './data/'  # Data subdirectory
    ]
    
    for location in common_locations:
        h5_files.extend(glob.glob(os.path.join(location, '*.h5')))
    
    # Remove duplicates and sort
    h5_files = sorted(list(set(h5_files)))
    
    # Return just the filenames, not the full paths
    return [os.path.basename(f) for f in h5_files]

# --- Create the Dash app ---
app = dash.Dash(__name__, 
                external_stylesheets=[
                    dbc.themes.BOOTSTRAP,
                    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"
                ],
                suppress_callback_exceptions=True)
server = app.server
server.secret_key = config.secret_key

# Disable the debug toolbar
app.enable_dev_tools(debug=False, dev_tools_ui=False, dev_tools_props_check=False)

# Create a directory for data files if it doesn't exist
os.makedirs(config.h5_files_dir, exist_ok=True)

# --- Define the app layout ---
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    # Main content div - will be updated based on the URL
    html.Div(id='page-content'),
    
    # Hidden divs to store state
    html.Div(id='data-store', style={'display': 'none'}),
    html.Div(id='page-store', children='1', style={'display': 'none'}),
    html.Div(id='selected-row', children='-1', style={'display': 'none'}),
    html.Div(id='max-row', children='0', style={'display': 'none'})
])

# --- Define the file selection layout ---
def create_file_selection_layout():
    h5_files = find_h5_files()
    
    return html.Div([
        html.Div([
            html.H1("H5 Data Visualization Dashboard", className="text-center mb-4"),
            
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Select a file to visualize", className="mb-0")
                ], className="bg-success text-white"),
                dbc.CardBody([
                    html.Div([
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.Div([
                                    html.I(className="bi bi-file-earmark-binary me-3", style={"fontSize": "1.5rem", "color": "#0d6efd"}),
                                    html.Span(file, className="fs-5"),
                                    html.I(className="bi bi-chevron-right ms-auto", style={"color": "#6c757d"})
                                ], className="d-flex align-items-center")
                            ], href=f"/load/{file}", action=True) 
                            for file in h5_files
                        ]) if h5_files else html.P("No H5 files found. Please add files to the data directory.", className="text-center")
                    ]),
                    
                    html.Hr(),
                    
                    dbc.Form([
                        dbc.InputGroup([
                            dbc.Input(id="file-name-input", placeholder="Enter file name (e.g., test.h5)", type="text"),
                            dbc.Button("Load", id="load-file-button", color="primary")
                        ])
                    ], className="mt-3")
                ])
            ], className="shadow")
        ], className="container py-5", style={"maxWidth": "800px"})
    ], style={"backgroundColor": "#f8f9fa", "minHeight": "100vh"})

# --- Define the visualization layout ---
def create_visualization_layout():
    return html.Div([
        html.H1("Your visualization", className="mb-4", 
               style={"backgroundColor": "#1e8449", "color": "white", "padding": "10px", "textAlign": "center"}),
        html.Button("BACK", id="back-button", className="btn btn-outline-secondary", style={"marginBottom": "20px", "marginLeft": "10px"}),
        
        # Simplified file info section
        html.Div(id='file-info', className="mb-3 alert alert-info py-2", style={"margin": "0 10px"}),
        
        dcc.Tabs(id='tabs', value='screwdown', children=[
            dcc.Tab(label='Screwdown', value='screwdown'),
            dcc.Tab(label='Bending Data', value='bending'),
            dcc.Tab(label='Profile Data', value='profile'),
            dcc.Tab(label='All Data', value='all_data'),
        ]),
        
        # Add a div for the table with a clear header
        html.Div([
            html.Div("Scroll horizontally to view more columns.", 
                    className="alert alert-info py-2", style={"margin": "10px 0"}),
            
            # Direct HTML table output - now using full width
            html.Div(id='table-container', style={"overflowX": "auto", "width": "100%"}),
            
            # Jump to blank info section
            html.Div([
                html.Label("Jump to blank info:", className="me-2", style={"fontWeight": "bold"}),
                dcc.Input(
                    id="jump-to-row-input",
                    type="number",
                    min=1,
                    className="form-control form-control-sm",
                    style={"width": "80px", "display": "inline-block", "marginRight": "10px"}
                ),
                html.Button(
                    "Go",
                    id="jump-to-row-button",
                    className="btn btn-primary btn-sm"
                )
            ], className="d-flex align-items-center justify-content-center mt-2 mb-4"),
            
            # Simple pagination controls - add initial buttons to ensure they exist
            html.Div([
                html.Button("Previous", id="prev-button", 
                          className="btn btn-outline-primary me-2",
                          disabled=True),
                html.Span("Page 1", className="mx-2"),
                html.Button("Next", id="next-button", 
                          className="btn btn-outline-primary ms-2")
            ], id='pagination-controls', className="d-flex justify-content-center mb-4"),
        ], className="mb-4"),

        # Empty div for graph that will be populated only when a row is selected
        html.Div(id='graph-container', style={"display": "none"}),
    ], style={"width": "100%", "padding": "0", "margin": "0"})  # Full width with no padding or margin

# --- Define the error layout ---
def create_error_layout(message):
    return html.Div([
        html.Div([
            html.Div([
                html.I(className="bi bi-exclamation-triangle-fill", 
                      style={"color": "#f39c12", "fontSize": "64px", "marginBottom": "20px"}),
                html.H1("Error", className="mb-4"),
                html.P(message, className="lead mb-4"),
                html.A("Back to Home", href="/", className="btn btn-primary")
            ], className="text-center py-5")
        ], className="container")
    ])

# --- Callback to update the page content based on URL ---
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/' or pathname == '/select_database':
        return create_file_selection_layout()
    elif pathname == '/visualize':
        if not data_store:
            return create_error_layout("No data available. Please select a file first.")
        return create_visualization_layout()
    elif pathname and pathname.startswith('/load/'):
        file_name = pathname.split('/load/')[1]
        
        # Check multiple possible locations for the file
        possible_paths = [
            os.path.join(config.h5_files_dir, file_name),
            file_name,  # Current directory
            f"./{file_name}",
            f"../{file_name}",
            f"./valid/{file_name}",
            f"./data/{file_name}"
        ]
        
        for file_path in possible_paths:
            if os.path.exists(file_path):
                if load_data_from_h5(file_path):
                    return dcc.Location(pathname="/visualize", id="redirect-to-visualize")
        
        return create_error_layout(f"Failed to load file: {file_name}")
    elif pathname == '/back':
        # Go back to previous file
        if len(file_history) > 1:
            # Remove current file from history
            file_history.pop()
            # Get previous file
            previous_file = file_history[-1]
            
            # Try to load the previous file
            possible_paths = [
                os.path.join(config.h5_files_dir, previous_file),
                previous_file,  # Current directory
                f"./{previous_file}",
                f"../{previous_file}",
                f"./valid/{previous_file}",
                f"./data/{previous_file}"
            ]
            
            for file_path in possible_paths:
                if os.path.exists(file_path):
                    if load_data_from_h5(file_path):
                        return dcc.Location(pathname="/visualize", id="redirect-to-visualize")
        
        return dcc.Location(pathname="/", id="redirect-to-home")
    else:
        return create_error_layout("Page not found")

# --- Callback for the load file button ---
@app.callback(
    Output('url', 'pathname'),
    [Input('load-file-button', 'n_clicks')],
    [State('file-name-input', 'value')]
)
def load_file_button(n_clicks, file_name):
    if n_clicks is None or not file_name:
        return dash.no_update
    
    return f"/load/{file_name}"

# --- Callback for the back button ---
@app.callback(
    Output('url', 'pathname', allow_duplicate=True),
    [Input('back-button', 'n_clicks')],
    prevent_initial_call=True
)
def back_button(n_clicks):
    if n_clicks is None:
        return dash.no_update
    
    return "/back"

# --- Callback to update file info ---
@app.callback(
    Output('file-info', 'children'),
    [Input('url', 'pathname')]
)
def update_file_info(pathname):
    # Get the current file
    if current_file_name:
        return html.Div([
            html.P(f"Currently viewing: ", className="mb-0"),
            html.Strong(current_file_name)
        ])
    else:
        return html.Div("No file selected", className="text-danger")

# --- Callback to update max row value ---
@app.callback(
    Output('max-row', 'children'),
    [Input('tabs', 'value')]
)
def update_max_row(tab):
    if not data_store:
        return "0"
    
    # For all_data tab, use screwdown data for row count
    if tab == 'all_data' and 'screwdown' in data_store:
        data = data_store['screwdown']
    elif tab not in data_store:
        return "0"
    else:
        data = data_store[tab]
    
    if 'actual_x' not in data or not hasattr(data['actual_x'], 'shape'):
        return "0"
        
    return str(data['actual_x'].shape[0])

# --- Callback to update page number and selected row ---
@app.callback(
    [Output('page-store', 'children'),
     Output('selected-row', 'children')],
    [Input('prev-button', 'n_clicks'),
     Input('next-button', 'n_clicks'),
     Input('jump-to-row-button', 'n_clicks'),
     Input('tabs', 'value')],
    [State('page-store', 'children'),
     State('jump-to-row-input', 'value'),
     State('selected-row', 'children'),
     State('max-row', 'children')]
)
def update_page_and_row(prev_clicks, next_clicks, jump_row_clicks, 
                       tab, current_page, jump_row, selected_row, max_row):
    ctx = callback_context
    if not ctx.triggered:
        return 1, -1
    
    # Reset page when tab changes
    if ctx.triggered[0]['prop_id'] == 'tabs.value':
        return 1, -1
    
    current_page = int(current_page) if current_page else 1
    max_row = int(max_row) if max_row else 0
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Default: don't change selected row
    new_selected_row = selected_row
    
    if button_id == 'prev-button' and prev_clicks:
        return max(1, current_page - 1), -1
    elif button_id == 'next-button' and next_clicks:
        # Calculate max pages (10 rows per page)
        max_pages = math.ceil(max_row / 10)
        return min(current_page + 1, max_pages), -1
    elif button_id == 'jump-to-row-button' and jump_row_clicks and jump_row:
        # Jump to the specified row
        # Convert to zero-based index
        row_index = int(jump_row) - 1
    
        # Validate row index
        if row_index < 0 or row_index >= max_row:
            # Invalid row, don't change anything
            return current_page, selected_row
        
        # Calculate which page this row would be on (5 rows per page)
        page = (row_index // 5) + 1
    
        # Return the page and the selected row
        return page, str(row_index)

    return current_page, new_selected_row

# --- Callback to handle the jump to row button ---
@app.callback(
    Output('jump-to-row-input', 'value', allow_duplicate=True),
    [Input('jump-to-row-button', 'n_clicks')],
    [State('jump-to-row-input', 'value'),
     State('max-row', 'children')],
    prevent_initial_call=True
)
def handle_jump_to_row(n_clicks, row_value, max_row):
    if n_clicks is None or n_clicks == 0 or row_value is None:
        return dash.no_update
    
    # Keep the input value after jumping (don't reset it)
    return row_value

# --- Callback to update pagination controls ---
@app.callback(
    Output('pagination-controls', 'children'),
    [Input('page-store', 'children'),
     Input('max-row', 'children')]
)
def update_pagination_controls(page, max_row):
    page = int(page) if page else 1
    max_row = int(max_row) if max_row else 0
    max_pages = math.ceil(max_row / 5) if max_row > 0 else 1  # Changed to 5 rows per page
    
    return html.Div([
        html.Button("Previous", id="prev-button", 
                  className="btn btn-outline-primary me-2",
                  disabled=page <= 1),
        html.Span(f"Page {page} of {max_pages}", className="mx-2"),
        html.Button("Next", id="next-button", 
                  className="btn btn-outline-primary ms-2",
                  disabled=page >= max_pages)
    ])

# --- Callback to update graph section ---
# @app.callback(
#     Output('graph-section', 'children'),
#     [Input('tabs', 'value'),
#      Input('selected-row', 'children')]
# )
# def update_graph_section(tab, selected_row):
#     # Check if a row is selected
#     if not selected_row or selected_row == '-1':
#         # Return empty div when no row is selected
#         return []
    
#     # Convert selected_row to int
#     selected_row = int(selected_row)
    
#     # Get title based on tab
#     title_map = {
#         'screwdown': 'Screwdown Graph',
#         'bending': 'Bending Graph',
#         'profile': 'Profile Graph',
#         'all_data': 'Combined Data Graphs'
#     }
#     title = title_map.get(tab, 'Data Graph')
#     title = f"{title} - Row {selected_row + 1}"
    
#     # Create graph based on tab and selected row
#     try:
#         if not data_store:
#             return html.Div("⚠️ No data for graph", className="alert alert-warning")
        
#         # For all_data tab, show a combined graph with all three data types
#         if tab == 'all_data':
#             # Check if we have all the necessary data
#             if not all(k in data_store for k in ['screwdown', 'bending', 'profile']):
#                 return html.Div("⚠️ Missing data for combined graph", className="alert alert-warning")
            
#             # Create a single Plotly figure for all data types
#             fig = go.Figure()
            
#             # Add Screwdown data
#             sd_data = data_store['screwdown']
#             if all(k in sd_data for k in ['actual_x', 'actual_z', 'ref_x', 'ref_z']):
#                 # Add reference data
#                 fig.add_trace(go.Scatter(
#                     x=sd_data['ref_x'],
#                     y=sd_data['ref_z'],
#                     mode='lines',
#                     name='Screwdown Ref',
#                     line=dict(color=REF_GRAPH_COLOR, width=4, dash='solid')
#                 ))
                
#                 # Add actual data
#                 if selected_row >= 0 and selected_row < sd_data['actual_x'].shape[0]:
#                     fig.add_trace(go.Scatter(
#                         x=sd_data['actual_x'][selected_row],
#                         y=sd_data['actual_z'][selected_row],
#                         mode='lines+markers',
#                         name='Screwdown Actual',
#                         line=dict(color=ACTUAL_GRAPH_COLOR, width=4, dash='solid'),
#                         marker=dict(size=12, color=ACTUAL_GRAPH_COLOR)
#                     ))
            
#             # Add Bending data
#             bd_data = data_store['bending']
#             if all(k in bd_data for k in ['actual_x', 'actual_z', 'ref_x', 'ref_z']):
#                 # Add reference data
#                 fig.add_trace(go.Scatter(
#                     x=bd_data['ref_x'],
#                     y=bd_data['ref_z'],
#                     mode='lines',
#                     name='Bending Ref',
#                     line=dict(color=REF_GRAPH_COLOR, width=4, dash='dash')
#                 ))
                
#                 # Add actual data
#                 if selected_row >= 0 and selected_row < bd_data['actual_x'].shape[0]:
#                     fig.add_trace(go.Scatter(
#                         x=bd_data['actual_x'][selected_row],
#                         y=bd_data['actual_z'][selected_row],
#                         mode='lines+markers',
#                         name='Bending Actual',
#                         line=dict(color=ACTUAL_GRAPH_COLOR, width=4, dash='dash'),
#                         marker=dict(size=12, color=ACTUAL_GRAPH_COLOR)
#                     ))
            
#             # Add Profile data
#             pd_data = data_store['profile']
#             if all(k in pd_data for k in ['actual_x', 'actual_z', 'ref_x', 'ref_z']):
#                 # Add reference data
#                 fig.add_trace(go.Scatter(
#                     x=pd_data['ref_x'],
#                     y=pd_data['ref_z'],
#                     mode='lines',
#                     name='Profile Ref',
#                     line=dict(color=REF_GRAPH_COLOR, width=4, dash='dot')
#                 ))
                
#                 # Add actual data
#                 if selected_row >= 0 and selected_row < pd_data['actual_x'].shape[0]:
#                     fig.add_trace(go.Scatter(
#                         x=pd_data['actual_x'][selected_row],
#                         y=pd_data['actual_z'][selected_row],
#                         mode='lines+markers',
#                         name='Profile Actual',
#                         line=dict(color=ACTUAL_GRAPH_COLOR, width=4, dash='dot'),
#                         marker=dict(size=12, color=ACTUAL_GRAPH_COLOR)
#                     ))
            
#             # Update layout
#             fig.update_layout(
#                 title=title,
#                 xaxis_title="X Position",
#                 yaxis_title="Z Position",
#                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
#                 margin=dict(l=40, r=40, t=40, b=40),
#                 height=600,
#                 hovermode="closest",
#                 plot_bgcolor="white",
#                 paper_bgcolor="white",
#                 font=dict(size=14)
#             )
            
#             # Add grid lines for better visibility
#             fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
#             fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
#             return [
#                 html.Div(title, className="alert alert-primary py-2 fw-bold", style={"margin": "0 10px"}),
#                 dcc.Graph(figure=fig)
#             ]
            
#         # For individual tabs, show a single graph
#         if tab not in data_store:
#             return html.Div(f"⚠️ No data for {tab} graph", className="alert alert-warning")

#         data = data_store[tab]
        
#         # Verify data has the required keys
#         if not all(k in data for k in ['actual_x', 'actual_z', 'ref_x', 'ref_z']):
#             return html.Div(f"⚠️ Missing data keys for {tab} graph", className="alert alert-warning")
        
#         # Verify data is not empty
#         if (len(data['ref_x'])== 0 or len(data['ref_z']) == 0 or 
#             data['actual_x'].size == 0 or data['actual_z'].size == 0):
#             return html.Div(f"⚠️ Empty data arrays for {tab} graph", className="alert alert-warning")
        
#         # Create a Plotly figure
#         fig = go.Figure()
    
#         # Add reference data
#         fig.add_trace(go.Scatter(
#             x=data['ref_x'],
#             y=data['ref_z'],
#             mode='lines',
#             name='Reference',
#             line=dict(color=REF_GRAPH_COLOR, width=4)
#         ))
    
#         # Add actual data for selected row
#         if selected_row >= 0 and selected_row < data['actual_x'].shape[0]:
#             fig.add_trace(go.Scatter(
#                 x=data['actual_x'][selected_row],
#                 y=data['actual_z'][selected_row],
#                 mode='lines+markers',
#                 name=f'Actual (Row {selected_row+1})',
#                 line=dict(color=ACTUAL_GRAPH_COLOR, width=4),
#                 marker=dict(size=12, color=ACTUAL_GRAPH_COLOR)
#             ))
#         else:
#             # If selected row is invalid, use the first row
#             fig.add_trace(go.Scatter(
#                 x=data['actual_x'][0],
#                 y=data['actual_z'][0],
#                 mode='lines+markers',
#                 name='Actual (Row 1)',
#                 line=dict(color=ACTUAL_GRAPH_COLOR, width=4),
#                 marker=dict(size=12, color=ACTUAL_GRAPH_COLOR)
#             ))
    
#         # Update layout
#         fig.update_layout(
#             title=title,
#             xaxis_title="X Position",
#             yaxis_title="Z Position",
#             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
#             margin=dict(l=40, r=40, t=40, b=40),
#             height=500,
#             plot_bgcolor="white",
#             paper_bgcolor="white",
#             font=dict(size=14)
#         )
        
#         # Add grid lines for better visibility
#         fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
#         fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
#         return [
#             html.Div(title, className="alert alert-primary py-2 fw-bold", style={"margin": "0 10px"}),
#             dcc.Graph(figure=fig)
#         ]

#     except Exception as e:
#         error_msg = f"❌ Error in graph: {str(e)}"
#         print(f"[ERROR] {error_msg}")
#         print(f"[ERROR] Traceback: {traceback.format_exc()}")
#         return html.Div([
#             html.P(error_msg, className="alert alert-danger"),
#             html.Pre(traceback.format_exc(), className="bg-light p-3 small")
#         ])

# --- Callback to handle radio button selection ---
@app.callback(
    Output('selected-row', 'children', allow_duplicate=True),
    [Input({'type': 'row-radio', 'index': dash.dependencies.ALL}, 'value')],
    prevent_initial_call=True
)
def update_selected_row_from_radio(values):
    ctx = callback_context
    if not ctx.triggered or not any(values):
        return dash.no_update
    
    # Find the selected value (should be only one)
    for value in values:
        if value is not None:
            return value
    
    return dash.no_update

# --- Callback to create a table with the format shown in the image ---
@app.callback(
    Output('table-container', 'children'),
    [Input('tabs', 'value'),
     Input('page-store', 'children'),
     Input('selected-row', 'children')]
)
def update_table(tab, page, selected_row):
    try:
        print(f"[DEBUG] Using colors: REF={REF_COLOR}, ACTUAL={ACTUAL_COLOR}")
        if not data_store:
            return html.Div("No data available for table", className="alert alert-warning")

        # Convert page and selected_row to int
        page = int(page) if page else 1
        selected_row = int(selected_row) if selected_row and selected_row != '-1' else -1

        # Get data for all three types
        sd_data = data_store.get('screwdown', {})
        bd_data = data_store.get('bending', {})
        pd_data = data_store.get('profile', {})

        # Check if we have the necessary data
        if not all(k in sd_data for k in ['actual_x', 'actual_z', 'ref_x', 'ref_z']):
            return html.Div("Missing screwdown data", className="alert alert-warning")
        if not all(k in bd_data for k in ['actual_x', 'actual_z', 'ref_x', 'ref_z']):
            return html.Div("Missing bending data", className="alert alert-warning")
        if not all(k in pd_data for k in ['actual_x', 'actual_z', 'ref_x', 'ref_z']):
            return html.Div("Missing profile data", className="alert alert-warning")

        # Get the number of rows
        max_rows = min(
            sd_data['actual_x'].shape[0],
            bd_data['actual_x'].shape[0],
            pd_data['actual_x'].shape[0]
        )

        # Calculate pagination
        rows_per_page = 5  # 5 rows per page (each row has REF and ACTUAL)
        
        # If a specific row is selected via jump-to-row, make sure it's visible
        if selected_row >= 0:
            # Calculate which page this row would be on
            page_for_selected_row = (selected_row // rows_per_page) + 1
            # Update the page to show the selected row
            page = page_for_selected_row
        
        start_idx = (page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, max_rows)

        # Create a completely different table layout using divs instead of HTML table
        # This approach gives more control over the layout and prevents overlapping
        
        # Create the table container
        table_container = html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "60px 100px 80px 1fr",
            "gap": "0px",
            "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
            "width": "max-content",
            "minWidth": "100%"
        })
        
        # Create header row
        header_row = html.Div(style={
            "display": "contents",
            "fontWeight": "bold",
            "backgroundColor": "#f8f9fa"
        })
        
        # Add header cells
        header_row.children = [
            html.Div("Index", style={
                "padding": "8px",
                "textAlign": "center",
                "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
                "backgroundColor": "white",
                "position": "sticky",
                "left": "0",
                "zIndex": "2"
            }),
            html.Div("Blank Info", style={
                "padding": "8px",
                "textAlign": "center",
                "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
                "backgroundColor": "white",
                "position": "sticky",
                "left": "60px",
                "zIndex": "2"
            }),
            html.Div("Values", style={
                "padding": "8px",
                "textAlign": "center",
                "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
                "backgroundColor": "white",
                "position": "sticky",
                "left": "160px",
                "zIndex": "2"
            })
        ]
        
        # Add data column headers
        if tab == 'all_data':
            # For All Data tab, show interleaved columns (SD1, BD1, PD1, SD2, BD2, PD2, etc.)
            data_headers = []
            
            # Determine how many sets of columns we need (max of SD, BD, PD points)
            max_points = max(len(sd_data['ref_x']), len(bd_data['ref_x']), len(pd_data['ref_x']))
            
            # Create interleaved headers
            for i in range(1, max_points + 1):
                # Add SD header
                data_headers.append(html.Div(f"SD{i}", style={
                    "padding": "8px",
                    "textAlign": "center",
                    "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
                    "minWidth": "120px"
                }))
                
                # Add BD header
                data_headers.append(html.Div(f"BD{i}", style={
                    "padding": "8px",
                    "textAlign": "center",
                    "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
                    "minWidth": "120px"
                }))
                
                # Add PD header
                data_headers.append(html.Div(f"PD{i}", style={
                    "padding": "8px",
                    "textAlign": "center",
                    "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
                    "minWidth": "120px"
                }))
            
            # Combine all headers into a single div
            data_headers_container = html.Div(data_headers, style={
                "display": "grid",
                "gridTemplateColumns": f"repeat({max_points * 3}, minmax(120px, 1fr))",
                "gap": "0px"
            })
            
            header_row.children.append(data_headers_container)
        else:
            # For individual tabs, show all columns for that type
            data = {'screwdown': sd_data, 'bending': bd_data, 'profile': pd_data}.get(tab, sd_data)
            prefix = {'screwdown': 'SD', 'bending': 'BD', 'profile': 'PD'}.get(tab, 'SD')
            
            data_headers = []
            for i in range(1, len(data['ref_x']) + 1):
                data_headers.append(html.Div(f"{prefix}{i}", style={
                    "padding": "8px",
                    "textAlign": "center",
                    "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
                    "minWidth": "120px"
                }))
                
            # Combine all headers into a single div
            data_headers_container = html.Div(data_headers, style={
                "display": "grid",
                "gridTemplateColumns": f"repeat({len(data['ref_x'])}, minmax(120px, 1fr))",
                "gap": "0px"
            })
            
            header_row.children.append(data_headers_container)
        
        # Add header row to table container
        table_container.children = [header_row]
        
        # Create data rows
        for i in range(start_idx, end_idx):
            # For each data row, we create two rows: one for REF and one for ACTUAL
            
            # Determine if this row is selected
            is_selected = (i == selected_row)
            
            # Create REF row with highlight if selected
            ref_row_style = {
                "display": "contents",
                "backgroundColor": REF_COLOR  # New color for REF rows
            }

            # Add highlight for selected row
            if is_selected:
                ref_row_style["backgroundColor"] = "#e6f7ff"  # Light blue highlight
            
            ref_row = html.Div(style=ref_row_style)
            
            # Add index cell with radio button (spans 2 rows)
            index_cell = html.Div([
                dcc.RadioItems(
                    id={'type': 'row-radio', 'index': i},
                    options=[{'label': '', 'value': str(i)}],
                    value=str(i) if is_selected else None,
                    style={'margin': '0', 'padding': '0'}
                )
            ], style={
                "padding": "8px",
                "textAlign": "center",
                "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
                "backgroundColor": "#e6f7ff" if is_selected else "white",  # Highlight if selected
                "position": "sticky",
                "left": "0",
                "zIndex": "1",
                "gridRow": f"span 2"
            })
            
            # Add blank info cell (spans 2 rows)
            blank_info_cell = html.Div(str(i + 1), style={
                "padding": "8px",
                "textAlign": "center",
                "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
                "backgroundColor": "#e6f7ff" if is_selected else "white",  # Highlight if selected
                "position": "sticky",
                "left": "60px",
                "zIndex": "1",
                "gridRow": f"span 2"
            })
            
            # Add REF cell
            ref_label_cell = html.Div(html.Span("REF", style={"color": "black", "fontWeight": "bold"}), style={
                "padding": "8px",
                "textAlign": "center",
                "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
                "backgroundColor": REF_COLOR if not is_selected else "#e6f7ff",  # Updated color for REF
                "position": "sticky",
                "left": "160px",
                "zIndex": "1"
            })
            
            # Add cells to REF row
            ref_row.children = [index_cell, blank_info_cell, ref_label_cell]
            
            # Create ACTUAL row with highlight if selected
            actual_row_style = {
                "display": "contents",
                "backgroundColor": ACTUAL_COLOR  # New color for ACTUAL rows
            }

            # Add highlight for selected row
            if is_selected:
                actual_row_style["backgroundColor"] = "#e6f7ff"  # Light blue highlight
            
            actual_row = html.Div(style=actual_row_style)
            
            # Add ACTUAL cell
            actual_label_cell = html.Div(html.Span("ACTUAL", style={"color": "black", "fontWeight": "bold"}), style={
                "padding": "8px",
                "textAlign": "center",
                "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
                "backgroundColor": ACTUAL_COLOR if not is_selected else "#e6f7ff",  # Updated color for ACTUAL
                "position": "sticky",
                "left": "160px",
                "zIndex": "1"
            })
            
            # Add cells to ACTUAL row
            actual_row.children = [actual_label_cell]
            
            # Add data cells based on the tab
            if tab == 'all_data':
                # For All Data tab, add all columns for all three data types
                
                # Create containers for data cells
                ref_data_cells = []
                actual_data_cells = []
                
                # Instead of adding all SD, then all BD, then all PD,
                # we'll interleave them as SD1, BD1, PD1, SD2, BD2, PD2, etc.
                
                # Determine how many sets of columns we need (max of SD, BD, PD points)
                max_points = max(len(sd_data['ref_x']), len(bd_data['ref_x']), len(pd_data['ref_x']))
                
                # Add interleaved SD, BD, PD data
                for j in range(max_points):
                    # Add SD reference cell
                    ref_x = sd_data['ref_x'][j]
                    ref_z = sd_data['ref_z'][j]
                    is_midpoint = sd_data['is_midpoint'][j] if j < len(sd_data['is_midpoint']) else False
                    # Display format depends on whether it's a midpoint
                    display_text = f"({int(ref_x)}, {int(ref_z)})" if not is_midpoint else f"({int(ref_x)}, -)"
                    ref_data_cells.append(html.Div(
                        display_text, 
                        style={
                            "padding": "8px",
                            "textAlign": "center",
                            "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
                            "color": "black",  # Change from blue to black
                            "minWidth": "120px",
                            "backgroundColor": REF_COLOR if not is_selected else "#e6f7ff"  # Updated color for REF
                        }
                    ))
                    
                    # Add SD actual cell
                    actual_x = sd_data['actual_x'][i][j] if i < len(sd_data['actual_x']) and j < sd_data['actual_x'].shape[1] else 0
                    actual_z = sd_data['actual_z'][i][j] if i < len(sd_data['actual_z']) and j < sd_data['actual_z'].shape[1] else 0
                    actual_data_cells.append(html.Div(
                        f"({int(actual_x)}, {int(actual_z)})", 
                        style={
                            "padding": "8px",
                            "textAlign": "center",
                            "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
                            "color": "black",  # Change from red to black
                            "minWidth": "120px",
                            "backgroundColor": ACTUAL_COLOR if not is_selected else "#e6f7ff"  # Updated color for ACTUAL
                        }
                    ))
                    
                    # Add BD reference cell
                    ref_x = bd_data['ref_x'][j]
                    ref_z = bd_data['ref_z'][j]
                    is_midpoint = bd_data['is_midpoint'][j] if j < len(bd_data['is_midpoint']) else False
                    # Display format depends on whether it's a midpoint
                    display_text = f"({int(ref_x)}, {int(ref_z)})" if not is_midpoint else f"({int(ref_x)}, -)"
                    ref_data_cells.append(html.Div(
                        display_text, 
                        style={
                            "padding": "8px",
                            "textAlign": "center",
                            "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
                            "color": "black",  # Change from blue to black
                            "minWidth": "120px",
                            "backgroundColor": REF_COLOR if not is_selected else "#e6f7ff"  # Updated color for REF
                        }
                    ))
                    
                    # Add BD actual cell
                    actual_x = bd_data['actual_x'][i][j] if i < len(bd_data['actual_x']) and j < bd_data['actual_x'].shape[1] else 0
                    actual_z = bd_data['actual_z'][i][j] if i < len(bd_data['actual_z']) and j < bd_data['actual_z'].shape[1] else 0
                    actual_data_cells.append(html.Div(
                        f"({int(actual_x)}, {int(actual_z)})", 
                        style={
                            "padding": "8px",
                            "textAlign": "center",
                            "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
                            "color": "black",  # Change from red to black
                            "minWidth": "120px",
                            "backgroundColor": ACTUAL_COLOR if not is_selected else "#e6f7ff"  # Updated color for ACTUAL
                        }
                    ))
                    
                    # Add PD reference cell
                    ref_x = pd_data['ref_x'][j]
                    ref_z = pd_data['ref_z'][j]
                    is_midpoint = pd_data['is_midpoint'][j] if j < len(pd_data['is_midpoint']) else False
                    # Display format depends on whether it's a midpoint
                    display_text = f"({int(ref_x)}, {int(ref_z)})" if not is_midpoint else f"({int(ref_x)}, -)"
                    ref_data_cells.append(html.Div(
                        display_text, 
                        style={
                            "padding": "8px",
                            "textAlign": "center",
                            "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
                            "color": "black",  # Change from blue to black
                            "minWidth": "120px",
                            "backgroundColor": REF_COLOR if not is_selected else "#e6f7ff"  # Updated color for REF
                        }
                    ))
                    
                    # Add PD actual cell
                    actual_x = pd_data['actual_x'][i][j] if i < len(pd_data['actual_x']) and j < pd_data['actual_x'].shape[1] else 0
                    actual_z = pd_data['actual_z'][i][j] if i < len(pd_data['actual_z']) and j < pd_data['actual_z'].shape[1] else 0
                    actual_data_cells.append(html.Div(
                        f"({int(actual_x)}, {int(actual_z)})", 
                        style={
                            "padding": "8px",
                            "textAlign": "center",
                            "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
                            "color": "black",  # Change from red to black
                            "minWidth": "120px",
                            "backgroundColor": ACTUAL_COLOR if not is_selected else "#e6f7ff"  # Updated color for ACTUAL
                        }
                    ))
                
                # Combine all data cells into containers
                # Calculate the total number of columns for the interleaved format
                max_points = max(len(sd_data['ref_x']), len(bd_data['ref_x']), len(pd_data['ref_x']))
                total_columns = max_points * 3  # SD, BD, PD for each point

                # Combine all data cells into containers
                ref_data_container = html.Div(ref_data_cells, style={
                    "display": "grid",
                    "gridTemplateColumns": f"repeat({total_columns}, minmax(120px, 1fr))",
                    "gap": "0px"
                })

                actual_data_container = html.Div(actual_data_cells, style={
                    "display": "grid",
                    "gridTemplateColumns": f"repeat({total_columns}, minmax(120px, 1fr))",
                    "gap": "0px"
                })
                
                # Add data containers to rows
                ref_row.children.append(ref_data_container)
                actual_row.children.append(actual_data_container)
            else:
                # For individual tabs, add all columns for that type
                data = {'screwdown': sd_data, 'bending': bd_data, 'profile': pd_data}.get(tab, sd_data)
                
                # Create containers for data cells
                ref_data_cells = []
                actual_data_cells = []
                
                for j in range(len(data['ref_x'])):
                    # Add reference cell
                    ref_x = data['ref_x'][j] if j < len(data['ref_x']) else 0
                    ref_z = data['ref_z'][j] if j < len(data['ref_z']) else 0
                    is_midpoint = data['is_midpoint'][j] if j < len(data['is_midpoint']) else False
                    # Display format depends on whether it's a midpoint
                    display_text = f"({int(ref_x)}, {int(ref_z)})" if not is_midpoint else f"({int(ref_x)}, -)"
                    ref_data_cells.append(html.Div(
                        display_text, 
                        style={
                            "padding": "8px",
                            "textAlign": "center",
                            "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
                            "color": "black",  # Change from blue to black
                            "minWidth": "120px",
                            "backgroundColor": REF_COLOR if not is_selected else "#e6f7ff"  # Updated color for REF
                        }
                    ))
                    
                    # Add actual cell
                    actual_x = data['actual_x'][i][j] if i < len(data['actual_x']) and j < data['actual_x'].shape[1] else 0
                    actual_z = data['actual_z'][i][j] if i < len(data['actual_z']) and j < data['actual_z'].shape[1] else 0
                    actual_data_cells.append(html.Div(
                        f"({int(actual_x)}, {int(actual_z)})", 
                        style={
                            "padding": "8px",
                            "textAlign": "center",
                            "border": f"1px solid {BORDER_COLOR}",  # Lighter border color
                            "color": "black",  # Change from red to black
                            "minWidth": "120px",
                            "backgroundColor": ACTUAL_COLOR if not is_selected else "#e6f7ff"  # Updated color for ACTUAL
                        }
                    ))
                
                # Combine all data cells into containers
                ref_data_container = html.Div(ref_data_cells, style={
                    "display": "grid",
                    "gridTemplateColumns": f"repeat({len(data['ref_x'])}, minmax(120px, 1fr))",
                    "gap": "0px"
                })
                
                actual_data_container = html.Div(actual_data_cells, style={
                    "display": "grid",
                    "gridTemplateColumns": f"repeat({len(data['ref_x'])}, minmax(120px, 1fr))",
                    "gap": "0px"
                })
                
                # Add data containers to rows
                ref_row.children.append(ref_data_container)
                actual_row.children.append(actual_data_container)
            
            # Add rows to table container
            table_container.children.extend([ref_row, actual_row])
        
        # Return the table container in a scrollable div
        return html.Div(
            table_container,
            style={
                "overflowX": "auto",
                "width": "100%",
                "maxWidth": "100vw",
                "paddingBottom": "15px",
                "maxHeight": "70vh",
                "overflowY": "auto",
                "margin": "0",
                "padding": "0"
            }
        )

    except Exception as e:
        error_msg = f"Error in table: {str(e)}"
        print(f"[ERROR] {error_msg}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return html.Div(error_msg, className="alert alert-danger")

# --- Callback to dynamically add or remove the graph section ---
# @app.callback(
#     Output('page-content', 'children'),
#     [Input('selected-row', 'children'),
#      Input('url', 'pathname')],
#     [State('tabs', 'value')]
# )
# def update_page_content_with_graph(selected_row, pathname, tab):
#     # Only update when we're on the visualization page
#     if pathname != '/visualize':
#         return dash.no_update
    
#     # Create the base visualization layout
#     layout = create_visualization_layout()
    
#     # Check if a row is selected
#     if selected_row and selected_row != '-1':
#         # Convert selected_row to int
#         selected_row = int(selected_row)
        
#         # Create graph section based on the selected row and tab
#         graph_section = create_graph_section(tab, selected_row)
        
#         # Add the graph section to the layout
#         layout.children.append(graph_section)
    
#     return layout

# Helper function to create the graph section
def create_graph_section(tab, selected_row):
    try:
        print(f"[DEBUG] Creating graph for tab={tab}, row={selected_row}")
        if not data_store:
            return html.Div("⚠️ No data for graph", className="alert alert-warning")
        
        # Get title based on tab
        title_map = {
            'screwdown': 'Screwdown Graph',
            'bending': 'Bending Graph',
            'profile': 'Profile Graph',
            'all_data': 'Combined Data Graphs'
        }
        title = title_map.get(tab, 'Data Graph')
        title = f"{title} - Row {selected_row + 1}"
        
        # For all_data tab, show a combined graph with all three data types
        if tab == 'all_data':
            # Check if we have all the necessary data
            if not all(k in data_store for k in ['screwdown', 'bending', 'profile']):
                return html.Div("⚠️ Missing data for combined graph", className="alert alert-warning")
            
            # Create a single Plotly figure for all data types
            fig = go.Figure()
            
            # Add Screwdown data
            sd_data = data_store['screwdown']
            if all(k in sd_data for k in ['actual_x', 'actual_z', 'ref_x', 'ref_z']):
                # Add reference data
                fig.add_trace(go.Scatter(
                    x=sd_data['ref_x'],
                    y=sd_data['ref_z'],
                    mode='lines',
                    name='Screwdown Ref',
                    line=dict(color=REF_GRAPH_COLOR, width=4, dash='solid')
                ))
                
                # Add actual data
                if selected_row >= 0 and selected_row < sd_data['actual_x'].shape[0]:
                    fig.add_trace(go.Scatter(
                        x=sd_data['actual_x'][selected_row],
                        y=sd_data['actual_z'][selected_row],
                        mode='lines+markers',
                        name='Screwdown Actual',
                        line=dict(color=ACTUAL_GRAPH_COLOR, width=4, dash='solid'),
                        marker=dict(size=12, color=ACTUAL_GRAPH_COLOR)
                    ))
            
            # Add Bending data
            bd_data = data_store['bending']
            if all(k in bd_data for k in ['actual_x', 'actual_z', 'ref_x', 'ref_z']):
                # Add reference data
                fig.add_trace(go.Scatter(
                    x=bd_data['ref_x'],
                    y=bd_data['ref_z'],
                    mode='lines',
                    name='Bending Ref',
                    line=dict(color=REF_GRAPH_COLOR, width=4, dash='dash')
                ))
                
                # Add actual data
                if selected_row >= 0 and selected_row < bd_data['actual_x'].shape[0]:
                    fig.add_trace(go.Scatter(
                        x=bd_data['actual_x'][selected_row],
                        y=bd_data['actual_z'][selected_row],
                        mode='lines+markers',
                        name='Bending Actual',
                        line=dict(color=ACTUAL_GRAPH_COLOR, width=4, dash='dash'),
                        marker=dict(size=12, color=ACTUAL_GRAPH_COLOR)
                    ))
            
            # Add Profile data
            pd_data = data_store['profile']
            if all(k in pd_data for k in ['actual_x', 'actual_z', 'ref_x', 'ref_z']):
                # Add reference data
                fig.add_trace(go.Scatter(
                    x=pd_data['ref_x'],
                    y=pd_data['ref_z'],
                    mode='lines',
                    name='Profile Ref',
                    line=dict(color=REF_GRAPH_COLOR, width=4, dash='dot')
                ))
                
                # Add actual data
                if selected_row >= 0 and selected_row < pd_data['actual_x'].shape[0]:
                    fig.add_trace(go.Scatter(
                        x=pd_data['actual_x'][selected_row],
                        y=pd_data['actual_z'][selected_row],
                        mode='lines+markers',
                        name='Profile Actual',
                        line=dict(color=ACTUAL_GRAPH_COLOR, width=4, dash='dot'),
                        marker=dict(size=12, color=ACTUAL_GRAPH_COLOR)
                    ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="X Position",
                yaxis_title="Z Position",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=40, b=40),
                height=600,
                hovermode="closest",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=14)
            )
            
            # Add grid lines for better visibility
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            return html.Div([
                html.Div(title, className="alert alert-primary py-2 fw-bold", style={"margin": "0 10px"}),
                dcc.Graph(figure=fig)
            ], className="mb-4")
            
        # For individual tabs, show a single graph
        if tab not in data_store:
            return html.Div(f"⚠️ No data for {tab} graph", className="alert alert-warning")

        data = data_store[tab]
        
        # Verify data has the required keys
        if not all(k in data for k in ['actual_x', 'actual_z', 'ref_x', 'ref_z']):
            return html.Div(f"⚠️ Missing data keys for {tab} graph", className="alert alert-warning")
        
        # Verify data is not empty
        if (len(data['ref_x'])== 0 or len(data['ref_z']) == 0 or 
            data['actual_x'].size == 0 or data['actual_z'].size == 0):
            return html.Div(f"⚠️ Empty data arrays for {tab} graph", className="alert alert-warning")
        
        # Create a Plotly figure
        fig = go.Figure()
    
        # Add reference data
        fig.add_trace(go.Scatter(
            x=data['ref_x'],
            y=data['ref_z'],
            mode='lines',
            name='Reference',
            line=dict(color=REF_GRAPH_COLOR, width=4)
        ))
    
        # Add actual data for selected row
        if selected_row >= 0 and selected_row < data['actual_x'].shape[0]:
            fig.add_trace(go.Scatter(
                x=data['actual_x'][selected_row],
                y=data['actual_z'][selected_row],
                mode='lines+markers',
                name=f'Actual (Row {selected_row+1})',
                line=dict(color=ACTUAL_GRAPH_COLOR, width=4),
                marker=dict(size=12, color=ACTUAL_GRAPH_COLOR)
            ))
        else:
            # If selected row is invalid, use the first row
            fig.add_trace(go.Scatter(
                x=data['actual_x'][0],
                y=data['actual_z'][0],
                mode='lines+markers',
                name='Actual (Row 1)',
                line=dict(color=ACTUAL_GRAPH_COLOR, width=4),
                marker=dict(size=12, color=ACTUAL_GRAPH_COLOR)
            ))
    
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="X Position",
            yaxis_title="Z Position",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=40),
            height=500,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=14)
        )
        
        # Add grid lines for better visibility
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
        return html.Div([
            html.Div(title, className="alert alert-primary py-2 fw-bold", style={"margin": "0 10px"}),
            dcc.Graph(figure=fig)
        ], className="mb-4")

    except Exception as e:
        error_msg = f"❌ Error in graph: {str(e)}"
        print(f"[ERROR] {error_msg}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return html.Div([
            html.P(error_msg, className="alert alert-danger"),
            html.Pre(traceback.format_exc(), className="bg-light p-3 small")
        ])

# --- Callback to dynamically add or remove the graph section ---
@app.callback(
    [Output('graph-container', 'children'),
     Output('graph-container', 'style')],
    [Input('tabs', 'value'),
     Input('selected-row', 'children')]
)
def update_graph_container(tab, selected_row):
    # Check if a row is selected
    if not selected_row or selected_row == '-1':
        # Return empty div when no row is selected and keep it hidden
        return [], {"display": "none"}
    
    # Convert selected_row to int
    selected_row = int(selected_row)
    
    # Create graph based on tab and selected row
    graph_content = create_graph_section(tab, selected_row)
    
    # Return the graph content and make it visible
    return graph_content, {"display": "block"}

# --- Main entry point ---
if __name__ == '__main__':
    # Find all available H5 files
    h5_files = find_h5_files()
    
    if h5_files:
        print(f"[INFO] Found {len(h5_files)} H5 files: {h5_files}")
        
        # Try to load each file until one succeeds
        loaded = False
        for file_name in h5_files:
            # Check multiple possible locations for the file
            possible_paths = [
                os.path.join(config.h5_files_dir, file_name),
                file_name,  # Current directory
                f"./{file_name}",
                f"../{file_name}",
                f"./valid/{file_name}",
                f"./data/{file_name}"
            ]
            
            for file_path in possible_paths:
                if os.path.exists(file_path):
                    print(f"[INFO] Attempting to load: {file_path}")
                    if load_data_from_h5(file_path):
                        print(f"[INFO] Successfully loaded data from {file_path}")
                        loaded = True
                        break
            
            if loaded:
                break
        
        if not loaded:
            print("[WARNING] Could not load any of the available H5 files.")
    else:
        print("[WARNING] No H5 files found.")

    # Run the app with debug mode disabled to remove the debug toolbar
    app.run(host='127.0.0.1', port=5000, debug=False)
