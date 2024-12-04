from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Camera, Lidar
#import cv2 not using camera currently
import numpy as np
from sklearn.cluster import DBSCAN
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured

scenario_map = {
    "smallgrid": {
        "mycar": {
            "position": (0, 0, 0),
            "rotation": (0, 0, 0, 1)
        },
        "aicar": {
            "position": (0, -150, 0),
            "rotation": (0, 0, 0, 1)
        },
        "aicar2": {
            "position": (10, -60, 0),
            "rotation": (0, 0, 0, 1)
        }
    },
    "gridmap_v2": {
        "mycar": {
            "position": (0, 0, 100),
            "rotation": (0, 0, 0, 1)
        },
        "aicar": {
            "position": (0, -10, 100),
            "rotation": (0, 0, 0, 1)
        }
    },
    "italy": {
        "mycar": {
            "position": (-353.763, 1169.096, 168.698),
            "rotation": (0, 0, 0.7071, 0.7071)
        },
        "aicar": {
            "position": (-361.763, 1169.096, 168.698),
            "rotation": (0, 0, 0.7071, 0.7071)
        }
    },
    "east_coast_usa": {
        "mycar": {
            "position": (598.1989509371833, -127.72784042535113, 53.44879727053376),
            "rotation": (0, 0, 1, 0)
        },
        "aicar": {
            "position": (600.1989509371833, -121.72784042535113, 53.44879727053376),
            "rotation": (0, 0, 1, 0)
        }
    }
}

# Function to start the scenario
def start_scenario(bng, scenario_map, map_name):
    scenario_config = scenario_map.get(map_name)
    if not scenario_config:
        raise ValueError(f"No configuration found for map: {map_name}")

    scenario = Scenario(map_name, 'AI_Car_Test')

    # Configure and add 'mycar'
    mycar_pos = scenario_config["mycar"]["position"]
    mycar_rot = scenario_config["mycar"]["rotation"]
    mycar = Vehicle('mycar', model='etk800', license='ITS', color='Red')
    scenario.add_vehicle(mycar, pos=mycar_pos, rot_quat=mycar_rot)

    # Configure and add 'ai_vehicle'
    # I realised we can add ai cars when simulation is running too, so this is not necessary
    '''ai_pos = scenario_config["aicar"]["position"]
    ai_rot = scenario_config["aicar"]["rotation"]
    ai_vehicle = Vehicle('ai_vehicle', model='etk800', license='AI_CAR', color='Blue')
    scenario.add_vehicle(ai_vehicle, pos=ai_pos, rot_quat=ai_rot)

    # Configure and add 'ai_vehicle'
    ai_pos = scenario_config["aicar2"]["position"]
    ai_rot = scenario_config["aicar2"]["rotation"]
    ai_vehicle2 = Vehicle('ai_vehicle2', model='etk800', license='AI_CAR', color='Green')
    scenario.add_vehicle(ai_vehicle2, pos=ai_pos, rot_quat=ai_rot)'''

    # Build and start the scenario
    scenario.make(bng)
    bng.settings.set_deterministic(60)
    bng.scenario.load(scenario)
    bng.scenario.start()

    # Change the speed unit to kph
    # mycar.control('unit_system', 'metric')

    return mycar #, ai_vehicle

def vehicle_location(vehicle: Vehicle):
        vehicle.sensors.poll()
        if vehicle.state:
            position = vehicle.state['pos']
            return position

def ground_removal(points, min_x, max_x, min_y, max_y, min_z, max_z, cell_size, tolerance):
    # filter out of range points
    in_bounds = (min_x <= points[:, 0]) & (points[:, 0] < max_x) & \
                (min_y <= points[:, 1]) & (points[:, 1] < max_y) & \
                (min_z <= points[:, 2]) & (points[:, 2] < max_z)
    
    points_filtered = points[in_bounds]

    # grid based on X and Y coordinates
    grid_width = int(np.ceil((max_x - min_x) / cell_size))
    grid_height = int(np.ceil((max_y - min_y) / cell_size))
    grid = np.full((grid_width, grid_height), np.inf)

    ## convert x and y coordinates into indexes
    xi = ((points_filtered[:, 0] - min_x) / cell_size).astype(np.int32)
    yi = ((points_filtered[:, 1] - min_y) / cell_size).astype(np.int32)
    zi = points_filtered[:, 2]

    # Sort points by Z (descending) so we can fill the grid with minimum Z values
    
    for x, y, z in zip(xi, yi, zi):
        grid[x, y] = np.minimum(grid[x, y], z)


    # create a mask to filter out ground points
    # make tolerance higher for slopey areas
    #adaptive_tolerance = 0.5 + 0.1 * np.abs(grid[xi, yi] - zi.mean())
    ground_mask = (zi <= (grid[xi, yi] + tolerance))

    # only non-ground points
    non_ground_points = points_filtered[~ground_mask]
    #non_ground_points = points_filtered[(~ground_mask) & (points_filtered[:, 2] < max_z )] 
    return non_ground_points


def stopping_dist(velocity, reaction_time = 1, friction_coef = 0.8, extra_dist = 1):
    #source: https://korkortonline.se/en/theory/reaction-braking-stopping/
    #Assuming by default we have near-perfect conditions with an extra metre to spare.
    reaction_dist = velocity * reaction_time / 3.6
    braking_dist = velocity**2 / 250 / friction_coef
    if velocity <= 10:
        extra_dist = 0
    return reaction_dist + braking_dist + extra_dist

def find_impact_zone(vehicle: Vehicle):
    #We define the impact zone as a rectangle ranging from the front bumper of the car up until stopping distance, while also being within limits of the vehicle's bounding box.
    #Note that the bounding box contains the min/max coordinates of the entire vehicle. 
    #This means that the vehicle losing a part like a mirror will cause the bounding box to "expand" while the vehicle moves as the mirror is left behind, but still counts as part of the box containing the vehicle.
    bbox = vehicle.get_bbox()
    #vehicle.update_vehicle()

    #velocity = vehicle.state['vel'] * 3.6 #velocity for all three axis in kph
    velocity = [v * 3.6 for v in vehicle.state['vel']]
    
    vel_3d = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2)**0.5

    #corners in front of the car
    fbl = bbox['front_bottom_left']
    fbr = bbox['front_bottom_right']
    ftl = bbox['front_top_left']
    ftr = bbox['front_top_right']
    
    stop_dist = stopping_dist(velocity = vel_3d)
    
    direction_vector = np.array(vehicle.state['vel']) / np.linalg.norm(vehicle.state['vel'])

    stop_vector = direction_vector * stop_dist
    #corners at the stopping distance
    
    sdbl = (fbl + stop_vector).tolist()
    sdbr = (fbr + stop_vector).tolist()
    sdtl = (ftl + stop_vector).tolist()
    sdtr = (ftr + stop_vector).tolist()

    return (list(fbl), list(fbr), list(ftl), list(ftr), sdbl, sdbr, sdtl, sdtr)

def is_point_in_impact_zone_old(point, impact_zone):
    x, y, z = point
    # Extract the bottom rectangle coordinates from the impact zone
    (fbl, fbr, ftl, ftr, sdbl, sdbr, sdtl, sdtr) = impact_zone
    # Check if the point is within the 2D bounds on the XY plane
    min_x = min(fbl[0], fbr[0], sdbl[0], sdbr[0])
    max_x = max(fbl[0], fbr[0], sdbl[0], sdbr[0])
    min_y = min(fbl[1], fbr[1], sdbl[1], sdbr[1])
    max_y = max(fbl[1], fbr[1], sdbl[1], sdbr[1])
    
    
    if (min_x <= x <= max_x) and (min_y <= y <= max_y):
        print(f"Bounds: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")
        print(f"Point: x={point[0]}, y={point[1]}")

    return (min_x <= x <= max_x) and (min_y <= y <= max_y)

from matplotlib.path import Path

def is_point_in_impact_zone(point, impact_zone):
    x, y, _ = point  # Ignore z for now
    polygon = Path([
        impact_zone[0][:2],  # fbl
        impact_zone[1][:2],  # fbr
        impact_zone[5][:2],  # sdbr
        impact_zone[4][:2],  # sdbl
    ])
    return polygon.contains_point((x, y))

def maintain_speed(mycar, speed_limit_kph):
    # Get the vehicle's speed in km/h
    velocity = mycar.state['vel']
    current_speed_kph = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2)**0.5 * 3.6  # Convert m/s to km/h

    # Adjust throttle and brake to maintain speed
    if current_speed_kph < speed_limit_kph:
        mycar.control(throttle=0.7, brake=0.0)  # Increase throttle if below speed limit
    else:
        mycar.control(throttle=0.0, brake=0.2)  # Apply brake if above speed limit

# apply the brake gradually not full brake
def gradual_brake(vehicle, current_brake, target_brake, step=0.05):
    if current_brake < target_brake:
        current_brake = min(current_brake + step, target_brake)
    elif current_brake > target_brake:
        current_brake = max(current_brake - step, target_brake)
    vehicle.control(brake=current_brake, throttle=0.0)
    return current_brake

def main():
    set_up_simple_logging()

    beamng_home = r'C:\\ITS\\BeamNG.tech.v0.32.5.0'
    user_folder = r'C:\\Users\\kaspe\\AppData\\Local\\BeamNG.drive\\0.32'

    bng = BeamNGpy('localhost', 64256, home=beamng_home, user=user_folder)
    bng.open(launch=True)

    # Start scenario based on the map data
    map_name = 'east_coast_usa'  # Change this to use a different map
    mycar = start_scenario(bng, scenario_map, map_name)

    # Create camera and attach it to vehicle
    '''camera = Camera(
        name='camera1',
        bng=bng,
        vehicle=mycar,
        is_streaming=True)'''


    lidar = Lidar(
        "lidar1",
        bng,
        mycar,
        requested_update_time=0.01,
        is_using_shared_memory=False,
        vertical_angle=40,
        horizontal_angle=60,
        vertical_resolution=64, #64
        #pos=(0, 0, 0), #pos=(0, -2.2, 0.7)
        #dir=(0, 0, 0),  
        is_360_mode=False,  # [DEMO: DEFAULT - 360 MODE].  Uses shared memory.
    )


    mycar.ai.set_mode('disabled')
    #mycar.ai_set_target(ai_vehicle)
    #mycar.ai.set_speed(50, 'set')
    #ai_vehicle.ai.set_mode('span')

    try:
        added_sphere_ids = []
        added_rect_ids = []
        current_brake=0
        while True:
            bng.control.step(1)
            
            mycar_xyz = vehicle_location(mycar)
            print("Car Location: " + str(mycar_xyz))
            mycar.sensors.poll()
            lidar_data = lidar.poll()

            points = lidar_data['pointCloud']

            #removing ground parameters
            min_x, max_x = mycar_xyz[0]-50,mycar_xyz[0]+50 # or -30, 30
            min_y, max_y = mycar_xyz[1]-50,mycar_xyz[1]+50 # -30, 30
            min_z, max_z = mycar_xyz[2],mycar_xyz[2]+2.5 # -2.5, 0.05
            cell_size = 1 # 0.6
            tolerance = 0.3 # 0.15
            


            filtered_points = ground_removal(
                points, min_x, max_x, min_y, max_y, min_z, max_z, cell_size, tolerance
            )
            
            if filtered_points.shape[0] < 1:
                print("No points remaining after ground removal.")
            else:
                # clustering the points
                clusterer = DBSCAN(eps=0.4, min_samples=2) # 0.7, 4
                labels = clusterer.fit_predict(filtered_points)

                unique_labels = np.unique(labels)
                centroids = []

                for label in unique_labels:
                    if label == -1:
                        continue
                    mask = (labels == label)
                    points3d = filtered_points[mask, :3]
                    if points3d.shape[0] < 4:
                        continue
                    centroid = points3d.mean(axis=0)
                    centroids.append(centroid)

                # Prepare data for adding debug spheres
                coordinates = [[float(centroid[0]), float(centroid[1]), float(centroid[2])] for centroid in centroids]
                radii = [float(0.3) for _ in centroids]
                colors = [(0, 1, 0, 1) for _ in centroids]  # Set color to green for all spheres (R, G, B, A)


                # All 8 corners for impact zone
                rec_coord = find_impact_zone(mycar)
                bottom = [
                        [rec_coord[0][0], rec_coord[0][1], rec_coord[0][2] + 0.2],
                        [rec_coord[1][0], rec_coord[1][1], rec_coord[1][2] + 0.2],
                        [rec_coord[5][0], rec_coord[5][1], rec_coord[5][2] + 0.2],
                        [rec_coord[4][0], rec_coord[4][1], rec_coord[4][2] + 0.2]
                    ]
                
                # Remove rectangles
                if added_rect_ids:
                    bng.remove_debug_rectangle(added_rect_ids)
                # Add debug rectangle only for the bottom
                added_rect_ids = bng.add_debug_rectangle(vertices=bottom, rgba_color=(1, 0, 0, 1)) # set color to red for all rectangles
                
                # Alternative method for visualizing the impact zone
                #for point in rec_coord:
                #    bng.add_debug_spheres([[point[0], point[1], point[2]]], [0.2], [(0, 0, 1, 1)])  # Blue spheres

                for i, centroid in enumerate(centroids):
                    if is_point_in_impact_zone(centroid, rec_coord):
                        print("Obstacle detected within the impact zone. Braking...")
                        #print(str(centroid))

                        colors[i] = (1, 0, 0, 1)
                        # Calculate distance to obstacle
                        dist_to_obstacle = np.linalg.norm(np.array(mycar_xyz) - np.array(centroid))
                        #print(dist_to_obstacle)
                        # Define a critical distance threshold for full braking
                        critical_distance = 15.0 
                        
                        if dist_to_obstacle <= critical_distance:
                            # Apply full brake if within the critical distance
                            target_brake = 1.0
                        else:
                            # Use a non-linear function for braking intensity
                            target_brake = min(1.0, max(0.1, (critical_distance / dist_to_obstacle) ** 4))
                        
                        # Apply the calculated brake intensity
                        mycar.control(brake=target_brake, throttle=0.0)
                        break  # Stop further checks once braking is applied
                    else:
                        # Release the brake if no obstacle is in the impact zone
                        mycar.control(brake=0.0)
                
                
                # Remove old spheres
                if added_sphere_ids:
                    bng.remove_debug_spheres(added_sphere_ids)
                added_sphere_ids.clear()

                # Add debug spheres to the simulator and store their IDs
                added_sphere_ids = bng.add_debug_spheres(coordinates=coordinates, radii=radii, rgba_colors=colors)
    except KeyboardInterrupt:
        print("Simulation stopped by the user.")
    finally:
        bng.close()

if __name__ == '__main__':
    main()