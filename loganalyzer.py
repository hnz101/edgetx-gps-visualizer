import pandas as pd
import numpy as np
from datetime import datetime
import json

# Custom JSON encoder to handle NumPy data types
class NumpyEncoder(json.JSONEncoder):
    """
    A JSON encoder that can handle NumPy-specific numeric types,
    converting them to standard Python types.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Helper functions to replace branca colormap functionality
def hex_to_rgb(hex_color):
    """Converts a hex color string like '#ffffff' to an (R, G, B) tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    """Converts an (R, G, B) tuple to a hex color string."""
    r, g, b = [max(0, min(255, int(c))) for c in rgb_color]
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def linear_interpolate(val, start, end):
    """Linearly interpolates a value between start and end."""
    return start + (end - start) * val

def get_color_from_gradient(value, vmin, vmax, colors_hex):
    """
    Calculates the color for a value based on a multi-color gradient.
    """
    if pd.isna(value) or vmin == vmax:
        return colors_hex[-1]
    if value <= vmin:
        return colors_hex[0]
    if value >= vmax:
        return colors_hex[-1]

    normalized_value = (value - vmin) / (vmax - vmin)
    num_segments = len(colors_hex) - 1
    segment_index = min(int(normalized_value * num_segments), num_segments - 1)
    segment_size = 1.0 / num_segments
    segment_proportion = (normalized_value - segment_index * segment_size) / segment_size
    start_color_rgb = hex_to_rgb(colors_hex[segment_index])
    end_color_rgb = hex_to_rgb(colors_hex[segment_index + 1])

    r = linear_interpolate(segment_proportion, start_color_rgb[0], end_color_rgb[0])
    g = linear_interpolate(segment_proportion, start_color_rgb[1], end_color_rgb[1])
    b = linear_interpolate(segment_proportion, start_color_rgb[2], end_color_rgb[2])

    return rgb_to_hex((r, g, b))

def find_column(df, possible_names):
    """Finds a column in the DataFrame from a list of possible names (case-insensitive)."""
    for name in possible_names:
        matches = [col for col in df.columns if name.lower() in col.lower()]
        if matches:
            return matches[0]
    return None

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the earth (specified in decimal degrees) in kilometers.
    """
    R = 6378  # Radius of Earth in kilometers
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def create_interactive_map(file_path, height_offset_threshold=50):
    """
    Reads flight data from a CSV, processes it, and creates an interactive map
    using Leaflet.js, saved as an HTML file.
    """
    print(f"Reading file: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.ParserError:
        raise pd.errors.ParserError(f"Error parsing file: {file_path}. Please check the format.")
    print(f"Available columns: {', '.join(df.columns)}")

    # --- Data Processing ---
    def extract_coords(coord_str):
        try:
            lat, lon = map(float, str(coord_str).split())
            return pd.Series({'latitude': lat, 'longitude': lon})
        except (ValueError, AttributeError):
            return pd.Series({'latitude': None, 'longitude': None})

    gps_column = find_column(df, ['gps', 'GPS', 'gps-coords', 'GPS-coords'])
    if not gps_column:
        raise ValueError("No GPS column found!")

    coords_df = df[gps_column].apply(extract_coords)
    df = pd.concat([df, coords_df], axis=1)
    df = df.dropna(subset=['latitude', 'longitude'])
    if df.empty:
        raise ValueError("No valid GPS data found after processing.")

    # --- Calculation for Info Box ---
    total_distance_km = 0
    average_speed_kmh = 0
    
    if len(df) > 1:
        # Vectorized haversine calculation for total distance
        lats = df['latitude'].to_numpy()
        lons = df['longitude'].to_numpy()
        distances = haversine(lats[:-1], lons[:-1], lats[1:], lons[1:])
        total_distance_km = np.sum(distances)

        # Calculate average speed if time data is available
        time_col = find_column(df, ['Time', 'zeit'])
        if time_col:
            df_time_sorted = df.copy()
            df_time_sorted['datetime'] = pd.to_datetime(df_time_sorted[time_col], errors='coerce')
            df_time_sorted = df_time_sorted.dropna(subset=['datetime']).sort_values('datetime')
            
            if len(df_time_sorted) > 1:
                duration = df_time_sorted['datetime'].iloc[-1] - df_time_sorted['datetime'].iloc[0]
                duration_hours = duration.total_seconds() / 3600
                
                if duration_hours > 0:
                    average_speed_kmh = total_distance_km / duration_hours

    info_box_data = {
        'total_distance': f"{total_distance_km:.2f} km",
        'average_speed': f"{average_speed_kmh:.2f} km/h" if average_speed_kmh > 0 else "N/A (no time data)"
    }
    
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()

    vis_types = {
        'speed': {
            'column': find_column(df, ['GSpd', 'Speed', 'Geschw']),
            'title': 'Geschwindigkeit (km/h)',
            'colors': ['#008000', '#ffff00', '#ff0000']
        },
        'altitude': {
            'column': find_column(df, ['Alt', 'GAlt', 'Altitude']),
            'title': 'Höhe (m)',
            'colors': ['#0000ff', '#00ff00', '#ff0000']
        },
        'rssi': {
            'column': find_column(df, ['RSSI', 'rssi', 'RSSi', 'RSS']),
            'title': 'RSSI (dBm)',
            'colors': ['#ff0000', '#ffff00', '#008000']
        }
    }

    layers_data = {}
    points = list(zip(df['latitude'], df['longitude']))

    for vis_type, config in vis_types.items():
        if config['column'] and config['column'] in df.columns:
            print(f"Creating visualization for {vis_type}")
            values = pd.to_numeric(df[config['column']], errors='coerce')

            if vis_type == 'altitude':
                diffs = np.abs(values.diff())
                stable_indices = diffs[diffs > height_offset_threshold].index
                if len(stable_indices) > 0:
                    first_stable_index = stable_indices[0]
                    filtered_values = values[first_stable_index:]
                    print(f"Altitude scale based on values from index {first_stable_index}.")
                else:
                    filtered_values = values
                    print("No significant altitude jumps found; using all values for scale.")
            else:
                filtered_values = values
            
            valid_filtered_values = filtered_values.dropna()
            vmin = valid_filtered_values.min() if not valid_filtered_values.empty else 0
            vmax = valid_filtered_values.max() if not valid_filtered_values.empty else 1
            
            segments = []
            for i in range(len(points) - 1):
                val = values.iloc[i]
                color = get_color_from_gradient(val, vmin, vmax, config['colors'])
                popup_content = f"{config['title']}: {val:.1f}" if pd.notna(val) else "No data"
                segments.append({
                    'coords': [points[i], points[i+1]],
                    'color': color,
                    'popup': popup_content
                })
            
            layers_data[vis_type] = {
                'title': config['title'],
                'segments': segments,
                'legend': {
                    'colors': config['colors'],
                    'min': round(vmin, 1),
                    'max': round(vmax, 1)
                }
            }

    start_marker = points[0] if points else None
    end_marker = points[-1] if points else None

    # --- HTML and JavaScript Generation ---
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flight Path Analysis</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
        <style>
            html, body {{ height: 100%; margin: 0; }}
            #map {{ height: 100%; width: 100%; }}
            .legend {{
                line-height: 18px;
                color: #333;
                background: white;
                padding: 6px 8px;
                border-radius: 5px;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
            }}
            .legend i {{
                width: 18px;
                height: 18px;
                float: left;
                margin-right: 8px;
                opacity: 0.9;
            }}
            .legend-title {{
                font-weight: bold;
                margin-bottom: 5px;
                text-align: center;
            }}
            .info-box {{
                background: rgba(255, 255, 255, 0.8);
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
                margin-top: 10px;
            }}
            .info-box h4 {{
                margin: 0 0 5px 0;
                text-align: center;
            }}
            .info-box p {{
                margin: 5px 0;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>

    <div id="map"></div>

    <script>
        // --- Map Initialization ---
        var map = L.map('map').setView([{center_lat}, {center_lon}], 14);

        // --- Tile Layers ---
        var osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }});
        var esriImagery = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
        }});
        osm.addTo(map);

        var baseMaps = {{
            "OpenStreetMap": osm,
            "Esri World Imagery": esriImagery
        }};
        
        // --- Data from Python ---
        var layersData = {json.dumps(layers_data, indent=4, cls=NumpyEncoder)};
        var startMarkerCoords = {json.dumps(start_marker, cls=NumpyEncoder)};
        var endMarkerCoords = {json.dumps(end_marker, cls=NumpyEncoder)};
        var infoBoxData = {json.dumps(info_box_data, cls=NumpyEncoder)};

        var overlayMaps = {{}};
        var titleToKeyMap = {{}};

        // --- Create Overlay Layers ---
        for (const key in layersData) {{
            const layerInfo = layersData[key];
            var featureGroup = L.layerGroup();
            
            layerInfo.segments.forEach(segment => {{
                L.polyline(segment.coords, {{
                    color: segment.color,
                    weight: 3,
                    opacity: 0.8
                }}).bindPopup(segment.popup).addTo(featureGroup);
            }});
            
            overlayMaps[layerInfo.title] = featureGroup;
            titleToKeyMap[layerInfo.title] = key;
        }}

        // Show the speed layer by default if it exists
        var initialLayerTitle = layersData.speed ? layersData.speed.title : null;
        if(initialLayerTitle && overlayMaps[initialLayerTitle]) {{
            overlayMaps[initialLayerTitle].addTo(map);
        }}

        // --- Info Box Control ---
        var infoBox = L.control({{position: 'topright'}});
        infoBox.onAdd = function (map) {{
            this._div = L.DomUtil.create('div', 'info-box');
            this.update();
            return this._div;
        }};
        infoBox.update = function () {{
            this._div.innerHTML = '<h4>Flight Summary</h4>' +
                '<p><strong>Total Distance:</strong> ' + (infoBoxData.total_distance || 'N/A') + '</p>' +
                '<p><strong>Average Speed:</strong> ' + (infoBoxData.average_speed || 'N/A') + '</p>';
        }};
        infoBox.addTo(map);

        // --- Layer Control ---
        L.control.layers(baseMaps, overlayMaps, {{collapsed: false}}).addTo(map);
        
        // --- Markers ---
        if (startMarkerCoords) {{
            L.marker(startMarkerCoords, {{title: "Start"}}).addTo(map).bindPopup("<b>Start</b>");
        }}
        if (endMarkerCoords) {{
            L.marker(endMarkerCoords, {{title: "End"}}).addTo(map).bindPopup("<b>End</b>");
        }}

        // --- Dynamic Legend ---
        var legend = L.control({{position: 'bottomright'}});
        legend.onAdd = function(map) {{
            this._div = L.DomUtil.create('div', 'info legend');
            this.update();
            return this._div;
        }};

        legend.update = function(layerKey) {{
            this._div.innerHTML = ''; // Clear previous legend
            if (!layerKey || !layersData[layerKey]) {{
                this._div.style.display = 'none';
                return;
            }}
            this._div.style.display = 'block';

            const legendInfo = layersData[layerKey].legend;
            const title = layersData[layerKey].title;
            const colors = legendInfo.colors;
            const gradient = colors.join(', ');

            this._div.innerHTML += `<div class="legend-title">${{title}}</div>`;
            this._div.innerHTML += 
                '<i style="background: linear-gradient(to right, ' + gradient + ');"></i> ' +
                `<span>${{legendInfo.min}}</span> &ndash; <span>${{legendInfo.max}}</span>`;
        }};
        legend.addTo(map);

        // Update legend when layer is added/removed
        var activeOverlayKey = initialLayerTitle ? titleToKeyMap[initialLayerTitle] : null;
        legend.update(activeOverlayKey); // Initial legend

        map.on('overlayadd', function(e) {{
            activeOverlayKey = titleToKeyMap[e.name];
            legend.update(activeOverlayKey);
        }});

        map.on('overlayremove', function(e) {{
            // If the removed layer was the active one, hide the legend
            if (activeOverlayKey === titleToKeyMap[e.name]) {{
                activeOverlayKey = null;
                legend.update(null);
            }}
        }});

    </script>
    </body>
    </html>
    """

    # Save the generated HTML to a file
    output_file = f'flight_map_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
    #with open(output_file, 'w', encoding='utf-8') as f:
    #   f.write(html_template)
    #print(f"Map has been saved as: {output_file}")
    
    return html_template, df