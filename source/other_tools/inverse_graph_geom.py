#! python3
#r: openpyxl, psycopg2, Ipython

import Rhino.Geometry as rg

# Inputs: 
# lines: A list of lines representing streets (input from Grasshopper)

network = lines

def invert_graph(network):
    # Step 1: Calculate midpoints
    midpoints = [line.PointAt(0.5) for line in network]
    
    # Step 2: Determine connections between streets (by checking shared endpoints)
    connections = []
    for i, line1 in enumerate(network):
        for j, line2 in enumerate(network):
            if i >= j:
                continue  # Avoid duplicate pairs and self-comparison
            
            # Check if the lines share an endpoint
            if line1.From == line2.From or line1.From == line2.To or \
               line1.To == line2.From or line1.To == line2.To:
                # Create a line between the midpoints of connected streets
                connections.append(rg.Line(midpoints[i], midpoints[j]))

    return midpoints, connections

# Call the function with your input list of lines
midpoints, connections = invert_graph(lines)

# Outputs for the Grasshopper component
inv_points = midpoints       # List of points (midpoints)
inv_lines = connections     # List of lines (connections between midpoints)
print(inv_points)
print(inv_lines)