from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Camera, Lidar
#import cv2 not using camera currently
import numpy as np
from sklearn.cluster import DBSCAN
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured

def ground_removal(points, min_x, max_x, min_y, max_y, min_z, max_z, cell_size, tolerance):
    # filter out of range points
    in_bounds = (min_x <= points[:, 0]) & (points[:, 0] < max_x) & \
                (min_y <= points[:, 1]) & (points[:, 1] < max_y) & \
                (min_z <= points[:, 2]) & (points[:, 2] < max_z)
    
    points_filtered = points[in_bounds]

    # grid based on X and Y coordinates
    grid_width = int(np.ceil((max_x - min_x) / cell_size))
    grid_height = int(np.ceil((max_y - min_y) / cell_size))

    grid = np.full((grid_width, grid_height), np.nan)

    ## convert x and y coordinates into indexes
    xi = ((points_filtered[:, 0] - min_x) / cell_size).astype(np.int32)
    yi = ((points_filtered[:, 1] - min_y) / cell_size).astype(np.int32)
    zi = points_filtered[:, 2]

    # Sort points by Z (descending) so we can fill the grid with minimum Z values
    sorted_idx = np.argsort(-zi)
    xi_sorted, yi_sorted, zi_sorted = xi[sorted_idx], yi[sorted_idx], zi[sorted_idx]

    # fill grid with minimum Z values
    grid[xi_sorted, yi_sorted] = zi_sorted

    # create a mask to filter out ground points
    ground_mask = (zi <= (grid[xi, yi] + tolerance))

    # only non-ground points
    non_ground_points = points_filtered[~ground_mask]

    return non_ground_points

def main():
    set_up_simple_logging()

    beamng_home = r'C:\\ITS\\BeamNG.tech.v0.32.5.0'
    user_folder = r'C:\\Users\\kaspe\\AppData\\Local\\BeamNG.drive\\0.32'

    bng = BeamNGpy('localhost', 64256, home=beamng_home, user=user_folder)
    bng.open(launch=True)

    scenario = Scenario('smallgrid', 'AI_Car_Test')

    mycar = Vehicle('mycar', model='etk800', license='ITS', color='Red')
    scenario.add_vehicle(mycar, pos=(0, 0, 0), rot_quat=(0, 0, 0, 1))

    ai_vehicle = Vehicle('ai_vehicle', model='etk800', license='AI_CAR', color='Blue')
    scenario.add_vehicle(ai_vehicle, pos=(0, -10, 0), rot_quat=(0, 0, 0, 1))  # Position it 10 meters in front

    scenario.make(bng)
    bng.settings.set_deterministic(60)
    bng.scenario.load(scenario)
    bng.scenario.start()

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
        vertical_angle=90,
        horizontal_angle=120,
        vertical_resolution=64,
        pos=(0, -2, 1),
        dir=(0, 0, 0),  
        is_360_mode=False,  # [DEMO: DEFAULT - 360 MODE].  Uses shared memory.
    )


    mycar.ai.set_mode('disabled') 
    ai_vehicle.ai.set_mode('span')

    try:
        added_sphere_ids = []

        while True:
            bng.control.step(1)

            lidar_data = lidar.poll()
            print(lidar_data['pointCloud'].shape)

            points = lidar_data['pointCloud']
            #print(raw_points.dtype.names)
            #points = structured_to_unstructured(raw_points[['x', 'y', 'z']], dtype=np.float32)

            #points_np = np.array(points[['x', 'y', 'z']])  # Convert structured array to numpy array

            #removing ground parameters
            min_x, max_x = -30, 70 # or -50,50
            min_y, max_y = -30, 30 # -50,50
            min_z, max_z = -2.5, 0.05 # -1.5,1.5
            cell_size = 0.6 # 0.5
            tolerance = 0.15 # 0.2

            filtered_points = ground_removal(
                points, min_x, max_x, min_y, max_y, min_z, max_z, cell_size, tolerance
            )
            
            if filtered_points.shape[0] < 1:
                print("No points remaining after ground removal.")
            else:
                # clustering the points
                clusterer = DBSCAN(eps=0.7, min_samples=4)
                labels = clusterer.fit_predict(filtered_points)

                if points.shape[0] == labels.shape[0]:
                    print("Points and labels are equal.")
                else:
                    print("The number of points does not match the number of labels.")

                # filtering noise and calculating centroids
                unique_labels = np.unique(labels)
                centroids = []

                for label in unique_labels:
                    if label == -1:
                        continue
                    mask = (labels == label)
                    points3d = points[mask, :3]
                    if points3d.shape[0] < 4:
                        continue
                    centroid = points3d.mean(axis=0)
                    centroids.append(centroid)
                    print(f"Centroid for cluster {label}: {centroid}")
                
                # Renove spheres
                if added_sphere_ids:
                    bng.remove_debug_spheres(added_sphere_ids)
                added_sphere_ids.clear()

                # Prepare data for adding debug spheres
                coordinates = [[float(centroid[0]), float(centroid[1]), float(centroid[2])] for centroid in centroids]
                radii = [float(0.3) for _ in centroids]
                colors = [(0, 1, 0, 1) for _ in centroids]  # Set color to green for all spheres (R, G, B, A)

                # Add debug spheres to the simulator and store their IDs
                added_sphere_ids = bng.add_debug_spheres(coordinates=coordinates, radii=radii, rgba_colors=colors)


            #if lidar_data is not None:
            #                # Process the LiDAR data here
            #                points = np.array(lidar_data['points'])
            #                print(f"LiDAR points: {points.shape}")

            #camera_data = camera.poll()

            '''if camera_data is not None and 'depth' in camera_data:
                # Convert the raw data into an image format
                img = np.array(camera_data['depth'], dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Display the image
                cv2.imshow('Camera View', img)
                
                # Break on key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break'''
    except KeyboardInterrupt:
        print("Simulation stopped by the user.")
    finally:
        bng.close()

if __name__ == '__main__':
    main()
