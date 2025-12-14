Satellite Orbit \& Pointing Visualization (MCP Server)

This project implements a Model Context Protocol (MCP) server that allows an AI assistant (like Claude) to launch 3D visualizations of satellites orbiting Earth. Specifically, it simulates the International Space Station (ISS) slewing its orientation to "stare" at a specific target location (latitude/longitude) on the ground as it passes by.



1\. Project Overview

The core function of this server is to translate a semantic request (e.g., "Show me the ISS looking at Tokyo") into a physics-based orbital simulation.



Input: Latitude and Longitude coordinates.



Orbital Mechanics: Uses live TLE (Two-Line Element) data to propagate the ISS orbit.



Visualization: Renders a 3D animation showing the Earth, the satellite's path, the ground target, and a dynamic "pointing vector" connecting the two.



2\. Architecture \& Tech Stack

This project is built using Python and follows the MCP standard to expose local Python functions to an LLM.



FastMCP: A high-level library that simplifies creating MCP servers. It handles the JSON-RPC communication between the host (Claude Desktop) and the Python script.



Skyfield: A pure Python astronomy library. It handles the heavy lifting of:



Loading orbital ephemeris data (TLE files).



Performing coordinate transformations (Earth-Centered Inertial to Earth-Centered Earth-Fixed).



Calculating precise positions of the satellite and the ground target over time.



Matplotlib: Used for the 3D rendering engine. It generates the wireframe Earth and handles the frame-by-frame animation of the orbital mechanics.



How it Fits Together

The User asks Claude: "Run a simulation over Paris."



Claude determines it needs the visualize\_orbit tool and looks up the coordinates for Paris.



MCP Client sends a request to our satellite\_server.py: call\_tool("visualize\_orbit", lat=48.85, lon=2.35).



Satellite Server receives the request and executes the simulation logic:



Downloads/Loads ISS orbital data.



Calculates the geometry for the next 95 minutes (one orbit).



Launches a Matplotlib window on the local machine.



The User sees the window pop up and watches the animation.



Satellite Server returns a success message to Claude ("Simulation completed...").



3\. Code Breakdown (satellite\_server.py)

The code is contained in a single file, split into two main parts: The MCP wrapper and the Simulation Logic.



Part A: The MCP Wrapper

This section initializes the FastMCP server. The @mcp.tool() decorator registers the function so it is visible to the AI client, handling the "API" contract.



Python



mcp = FastMCP("Satellite Sim")



@mcp.tool()

def visualize\_orbit(latitude: float, longitude: float) -> str:

&nbsp;   # ... calls run\_simulation\_logic(latitude, longitude)

&nbsp;   ...

Part B: The Simulation Logic (run\_simulation\_logic)

This function performs the physics simulation in four steps:



Data Loading: Loads TLEs from the bundled `stations.txt` by default (override with `SAT_ORBIT_TLE_SOURCE`, which may be a local path or a URL), and selects the ISS object.



Time Propagation: Creates a time array spanning approximately one ISS orbital period (95 minutes).



Coordinate Transformation (The "Hard" Math): Calculates the satellite's position in GCRS (inertial frame) and the target's dynamic position in the same frame (since the Earth rotates).



Animation Loop (update function): Iterates through the calculated positions to update the 3D plot elements (orbit path, satellite marker, target marker, and pointing vector) for each frame.



4\. Setup \& Usage

Prerequisites

Install the required Python packages:



Bash



pip install fastmcp skyfield matplotlib numpy

Configuration

Add the server to your Claude Desktop configuration file (claude\_desktop\_config.json):



Example Configuration (adjust path as needed):



JSON



{

&nbsp; "mcpServers": {

&nbsp;   "satellite-sim": {

&nbsp;     "command": "python",

&nbsp;     "args": \["/path/to/your/satellite\_server.py"]

&nbsp;   }

&nbsp; }

}

Running It

Restart Claude Desktop.



The server starts automatically in the background.



Prompt Claude: "Visualize a satellite pass over \[City Name]."



5\. Troubleshooting

"List indices must be integers": This was corrected by ensuring the TLE data (a list) is properly converted to a dictionary for name lookup.



Window not appearing: Ensure matplotlib and all dependencies are installed in the Python environment used by the server.



Blocking: The current implementation uses plt.show(), which blocks the server response until the visualization window is manually closed. This is a current limitation.

