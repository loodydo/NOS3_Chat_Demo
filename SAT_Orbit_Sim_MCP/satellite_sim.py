import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skyfield.api import load, wgs84
from matplotlib.animation import FuncAnimation

# --- Configuration ---
# Target: New York City (Lat: 40.7128 N, Long: 74.0060 W)
TARGET_LAT = 40.7128
TARGET_LON = -74.0060
SIMULATION_DURATION_MINS = 95  # Roughly one orbit
FRAMES = 200

def run_simulation():
    # 1. Load Ephemeris Data
    print("Loading orbital data...")
    ts = load.timescale()
    
    # Load the ISS TLE data
    stations_url = 'http://celestrak.org/NORAD/elements/stations.txt'
    satellites_list = load.tle_file(stations_url)
    
    # --- FIX: Convert list to dictionary for name lookup ---
    satellites = {sat.name: sat for sat in satellites_list}
    
    # Check if 'ISS (ZARYA)' exists, otherwise try just 'ISS'
    if 'ISS (ZARYA)' in satellites:
        satellite = satellites['ISS (ZARYA)']
    elif 'ISS' in satellites:
        satellite = satellites['ISS']
    else:
        # Fallback: just grab the first one if ISS isn't found
        print("ISS not found exactly, using first satellite in list.")
        satellite = satellites_list[0]
        
    print(f"Loaded satellite: {satellite.name}")

    # 2. Setup Time Range
    t0 = ts.now()
    minutes = np.linspace(0, SIMULATION_DURATION_MINS, FRAMES)
    # Build time array
    # Skyfield's ts.utc accepts arrays of minutes
    times = ts.utc(t0.utc_datetime().year, 
                   t0.utc_datetime().month, 
                   t0.utc_datetime().day, 
                   t0.utc_datetime().hour, 
                   t0.utc_datetime().minute + minutes)

    # 3. Calculate Positions
    # Get Geocentric positions
    geocentric = satellite.at(times)
    
    # Extract Satellite X, Y, Z coordinates (GCRS) in km
    x_sat, y_sat, z_sat = geocentric.position.km

    # Calculate Target Position in GCRS
    # We must calculate this for *each time step* because Earth rotates relative to GCRS
    target_loc = wgs84.latlon(TARGET_LAT, TARGET_LON)
    target_positions = target_loc.at(times).position.km
    x_targ, y_targ, z_targ = target_positions

    # 4. Visualization Setup
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw Earth (Wireframe sphere)
    R_earth = 6371.0
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_earth = R_earth * np.cos(u) * np.sin(v)
    y_earth = R_earth * np.sin(u) * np.sin(v)
    z_earth = R_earth * np.cos(v)
    ax.plot_wireframe(x_earth, y_earth, z_earth, color='gray', alpha=0.3)

    # Plot lines and points
    orbit_line, = ax.plot([], [], [], 'b--', label='Orbit Path', alpha=0.5)
    sat_point, = ax.plot([], [], [], 'ro', label='Satellite')
    target_point, = ax.plot([], [], [], 'g^', label='Target')
    pointing_vector, = ax.plot([], [], [], 'r-', lw=2, label='Pointing Vector')

    # Status Text
    status_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    # Set bounds
    max_range = R_earth + 2000
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.legend()
    ax.set_title(f"Satellite Tracking Target ({TARGET_LAT}, {TARGET_LON})")

    def update(frame):
        # Current Satellite Position
        cx, cy, cz = x_sat[frame], y_sat[frame], z_sat[frame]
        
        # Current Target Position
        tx, ty, tz = x_targ[frame], y_targ[frame], z_targ[frame]

        # Update Orbit Trace
        orbit_line.set_data(x_sat[:frame], y_sat[:frame])
        orbit_line.set_3d_properties(z_sat[:frame])

        # Update Satellite Marker
        sat_point.set_data([cx], [cy])
        sat_point.set_3d_properties([cz])

        # Update Target Marker
        target_point.set_data([tx], [ty])
        target_point.set_3d_properties([tz])

        # Update Pointing Vector (Line from Sat to Target)
        pointing_vector.set_data([cx, tx], [cy, ty])
        pointing_vector.set_3d_properties([cz, tz])
        
        # Calculate distance
        dist = np.sqrt((tx-cx)**2 + (ty-cy)**2 + (tz-cz)**2)
        
        status_text.set_text(f"Frame: {frame}/{FRAMES}\nDist to Target: {dist:.1f} km")

        return orbit_line, sat_point, target_point, pointing_vector, status_text

    print("Starting animation...")
    ani = FuncAnimation(fig, update, frames=FRAMES, interval=50, blit=False)
    plt.show()

if __name__ == "__main__":
    run_simulation()
