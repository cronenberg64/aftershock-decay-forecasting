import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')

# Ensure directories exist
os.makedirs(DATA_RAW, exist_ok=True)

MAINSHOCKS = [
    {
        'name': 'Ridgecrest_2019',
        'lat': 35.7695,
        'lon': -117.5993,
        'time': '2019-07-06T03:19:53',
        'magnitude': 7.1,
        'search_radius_deg': 1.5,
        'completeness_magnitude': 2.5
    },
    {
        'name': 'Northridge_1994',
        'lat': 34.213,
        'lon': -118.537,
        'time': '1994-01-17T12:30:55',
        'magnitude': 6.7,
        'search_radius_deg': 1.0,
        'completeness_magnitude': 2.5
    },
    {
        'name': 'Landers_1992',
        'lat': 34.200,
        'lon': -116.437,
        'time': '1992-06-28T11:57:34',
        'magnitude': 7.3,
        'search_radius_deg': 2.0,
        'completeness_magnitude': 2.5
    },
    {
        'name': 'Hector_Mine_1999',
        'lat': 34.594,
        'lon': -116.271,
        'time': '1999-10-16T09:46:44',
        'magnitude': 7.1,
        'search_radius_deg': 1.5,
        'completeness_magnitude': 2.5
    },
    {
        'name': 'Loma_Prieta_1989',
        'lat': 37.040,
        'lon': -121.877,
        'time': '1989-10-18T00:04:15',
        'magnitude': 6.9,
        'search_radius_deg': 1.5,
        'completeness_magnitude': 2.5
    },
    {
        'name': 'Tohoku_2011',
        'lat': 38.297,
        'lon': 142.373,
        'time': '2011-03-11T05:46:24',
        'magnitude': 9.0,
        'search_radius_deg': 5.0,
        'completeness_magnitude': 4.5
    },
    {
        'name': 'Kumamoto_2016',
        'lat': 32.791,
        'lon': 130.754,
        'time': '2016-04-15T16:25:06',
        'magnitude': 7.0,
        'search_radius_deg': 1.5,
        'completeness_magnitude': 4.5
    },
    {
        'name': 'Kobe_1995',
        'lat': 34.583,
        'lon': 135.011,
        'time': '1995-01-16T20:46:52',
        'magnitude': 6.9,
        'search_radius_deg': 1.0,
        'completeness_magnitude': 4.5
    },
    {
        'name': 'Maule_2010',
        'lat': -36.122,
        'lon': -72.898,
        'time': '2010-02-27T06:34:11',
        'magnitude': 8.8,
        'search_radius_deg': 4.0,
        'completeness_magnitude': 4.5
    }
]

def fetch_data():
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    manifest_lines = [
        "# Aftershock Data Fetch Manifest",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "| Mainshock | Start Time | End Time | Lat | Lon | Radius (deg) | Min Mag | Rows | Status |",
        "|---|---|---|---|---|---|---|---|---|"
    ]
    
    failed_queries = []
    
    for ms in MAINSHOCKS:
        name = ms['name']
        start_time = datetime.fromisoformat(ms['time'])
        end_time = start_time + timedelta(days=90)
        
        # We query for ALL events > regional completeness.
        # This will include the mainshock if it falls in the window/magnitude range.
        params = {
            'format': 'csv',
            'starttime': start_time.isoformat(),
            'endtime': end_time.isoformat(),
            'latitude': ms['lat'],
            'longitude': ms['lon'],
            'maxradius': ms['search_radius_deg'],
            'minmagnitude': ms['completeness_magnitude']
        }
        
        print(f"Fetching data for {name}...")
        
        success = False
        for attempt in range(3):
            try:
                response = requests.get(base_url, params=params, timeout=30)
                if response.status_code == 200:
                    csv_path = os.path.join(DATA_RAW, f"{name}_aftershocks.csv")
                    with open(csv_path, 'wb') as f:
                        f.write(response.content)
                    
                    df = pd.read_csv(csv_path)
                    rows = len(df)
                    
                    manifest_lines.append(
                        f"| {name} | {start_time.isoformat()} | {end_time.isoformat()} | "
                        f"{ms['lat']} | {ms['lon']} | {ms['search_radius_deg']} | "
                        f"{ms['completeness_magnitude']} | {rows} | SUCCESS |"
                    )
                    print(f"  Success: {rows} events fetched.")
                    success = True
                    break
                else:
                    print(f"  Attempt {attempt+1} failed with status {response.status_code}.")
            except Exception as e:
                print(f"  Attempt {attempt+1} failed with error: {e}")
            
            time.sleep(2) # wait before retry
            
        if not success:
            print(f"  FAILED to fetch {name} after 3 attempts.")
            failed_queries.append(name)
            manifest_lines.append(
                f"| {name} | {start_time.isoformat()} | {end_time.isoformat()} | "
                f"{ms['lat']} | {ms['lon']} | {ms['search_radius_deg']} | "
                f"{ms['completeness_magnitude']} | N/A | FAILED |"
            )
            
    with open(os.path.join(DATA_RAW, 'MANIFEST.md'), 'w') as f:
        f.write('\n'.join(manifest_lines))
        
    print("\n--- Summary ---")
    if failed_queries:
        print(f"Failed queries: {', '.join(failed_queries)}")
    else:
        print("All queries successful.")

if __name__ == "__main__":
    fetch_data()
