import numpy as np
import cv2
from matplotlib import pyplot as plt
from m6bk import *

import sys

np.set_printoptions(precision=2, threshold=sys.maxsize)

np.random.seed(1)

print(" \nTo Work: images like 0.png, 1.png must be in data/rgb folder.")
print ("To Work: depth maps for every image, like 0.dat, 1.dat must be in data/depth folder.")
print("To Work: segmentation output of image, like 0.dat, 1.dat must be in data/segmentation folder.")
print("To Work: object_detection of images, k and etc. are hard coded in DatasetHandler Class from m6bk.py\n")
print("The dataset handler contains three test data frames 0, 1, and 2. Each frames contains:")
print("DatasetHandler().rgb: a camera RGB image.")
print("DatasetHandler().depth: a depth image containing the depth in meters for every pixel.")
print("DatasetHandler().segmentation: an image containing the output of a semantic segmentation neural network as the category per pixel.")
print("DatasetHandler().object_detection: a numpy array containing the output of an object detection network.\n")

dataset_handler = DatasetHandler()

# Current Frame 1.png
dataset_handler.set_frame(1)

# The current data frame being read can be known through the following line of code:
print("The Current Frame:\n " + str(dataset_handler.current_frame))

image=dataset_handler.image

k = dataset_handler.k
print("\nK Matrix:\n" + str(k))

depth = dataset_handler.depth

_, image_cells = plt.subplots(1, 2, figsize=(10, 10))
image_cells[0].imshow(image)
image_cells[0].set_title('Current Frame')
image_cells[1].imshow(depth, cmap='jet')
image_cells[1].set_title('Depth Map of Current Frame')
plt.show()




segmentation = dataset_handler.segmentation

# Category	Mapping Index	Visualization Color: 
# Background	  0	  Black         Buildings	1	Red    Pedestrians	4	Teal     Poles	5	White
# Lane Markings	6	Purple      Roads	7	Blue       Side Walks	8	Yellow   Vehicles	10	Green
colored_segmentation = dataset_handler.vis_segmentation(segmentation)

_, image_cells = plt.subplots(1, 2, figsize=(10, 10))
image_cells[0].imshow(segmentation)
image_cells[0].set_title('Segmentation')
image_cells[1].imshow(colored_segmentation)
image_cells[1].set_title('Colored Segmentation')
plt.show()



### I - Drivable Space Estimation Using Semantic Segmentation Output
def xy_from_depth(depth, k):
    """
    Computes the x, and y coordinates of every pixel in the image using the depth map and the calibration matrix.

    Arguments:
    depth -- tensor of dimension (H, W), contains a depth value (in meters) for every pixel in the image.
    k -- tensor of dimension (3x3), the intrinsic camera matrix

    Returns:
    x -- tensor of dimension (H, W) containing the x coordinates of every pixel in the camera coordinate frame.
    y -- tensor of dimension (H, W) containing the y coordinates of every pixel in the camera coordinate frame.
    """
    
    # Gets the shape of the depth tensor.
    H, W=depth.shape
    print("Depth Shape (H, W):")
    print(H, W)
    
    # Grabs required parameters from the K matrix.
    f=k[0][0]
    u_o=k[0][2]
    v_o=k[1][2]
    
    # Generates a grid of coordinates corresponding to the shape of the depth map.
    x=np.zeros((H,W))
    y=np.zeros((H,W))
    
    # Computes x and y coordinates.
    for i in range (0, H):
        for j in range (0, W):
            x[i,j]=(j+1-u_o)*depth[i,j]/f
            y[i,j]=(i+1-v_o)*depth[i,j]/f
    
    return x, y





k = dataset_handler.k

z = dataset_handler.depth

x, y = xy_from_depth(z, k)



# Gets road mask by choosing pixels in segmentation output with value 7.
road_mask = np.zeros(segmentation.shape)
road_mask[segmentation == 7] = 1

plt.title('Road Mask')
plt.imshow(road_mask)

# Gets x,y, and z coordinates of pixels in road mask.
x_ground = x[road_mask == 1]
y_ground = y[road_mask == 1]
z_ground = dataset_handler.depth[road_mask == 1]
xyz_ground = np.stack((x_ground, y_ground, z_ground))




def ransac_plane_fit(xyz_data):
    """
    Computes plane coefficients a,b,c,d of the plane in the form ax+by+cz+d = 0
    using ransac for outlier rejection.

    Arguments:
    xyz_data -- tensor of dimension (3, N), contains all data points from which random sampling will proceed.
    num_itr -- 
    distance_threshold -- Distance threshold from plane for a point to be considered an inlier.

    Returns:
    p -- tensor of dimension (1, 4) containing the plane parameters a,b,c,d
    """

    N=xyz_data.shape[1]
    
    # Sets thresholds:
    num_itr = 10  # RANSAC maximum number of iterations.
    min_num_inliers = N - 0.01*N   # RANSAC minimum number of inliers.
    distance_threshold = 0.001  # Maximum distance from point to plane for point to be considered inlier.
    
    for i in range(num_itr):
        # Step 1: Choose a minimum of 3 points from xyz_data at random.
        points_randomindex = np.random.choice(N, 5)
        points_random = xyz_data[:,points_randomindex]
        
        # Step 2: Compute plane model.
        p=compute_plane(points_random)
        
        # Step 3: Find number of inliers.
        pr = p.reshape((4,1))
        xr = xyz_data[0,:].reshape((N,1))
        yr = xyz_data[1,:].reshape((N,1))

        zr = xyz_data[2,:].reshape((N,1))
        
        dst = np.abs(dist_to_plane(pr, xr, yr, zr))

        numinliers = sum(dst < distance_threshold)
        
        # Step 4: Checks if the current number of inliers is greater than all previous iterations and keep the inlier set with the largest number of points.
        if numinliers > 1000:       #min_num_inliers:
            min_num_inlierss = numinliers
            xkeep = xr[dst < distance_threshold]
            ykeep = yr[dst < distance_threshold]
            zkeep = zr[dst < distance_threshold]
            xyzkeep = np.stack((xkeep, ykeep, zkeep))
           
        # Step 5: Checks if stopping criterion is satisfied and break.         
            break
        
    # Step 6: Recomputes the model parameters using largest inlier set.         
    output_plane = compute_plane(xyzkeep)
    
    return output_plane 


p_final = ransac_plane_fit(xyz_ground)
print('Ground Plane: ' + str(p_final))



dist = np.abs(dist_to_plane(p_final, x, y, z))

ground_mask = np.zeros(dist.shape)

ground_mask[dist < 0.1] = 1
ground_mask[dist > 0.1] = 0






### II - Lane Estimation Using The Semantic Segmentation Output
def estimate_lane_lines(segmentation_output):
    """
    Estimates lines belonging to lane boundaries. Multiple lines could correspond to a single lane.

    Arguments:
    segmentation_output -- tensor of dimension (H,W), containing semantic segmentation neural network output
    minLineLength -- Scalar, the minimum line length
    maxLineGap -- Scalar, dimension (Nx1), containing the z coordinates of the points

    Returns:
    lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2], where [x_1,y_1] and [x_2,y_2] are
    the coordinates of two points on the line in the (u,v) image coordinate frame.
    """

    # Step 1: Creates an image with pixels belonging to lane boundary categories from the output of semantic segmentation.
    lane_mask= np.zeros(segmentation_output.shape)
    lane_mask[np.logical_or(segmentation_output==6, segmentation_output==8)]=1

    # Step 2: Performs Edge Detection using cv2.Canny()
    med_val = np.median(lane_mask) 
    
    lower1 = int(max(0, 0.7* med_val))
    upper1 = int(min(255,1.3 * med_val))
    
    lane_mask=lane_mask.astype(np.uint8)
    
    edges= cv2.Canny(lane_mask, threshold1=0, threshold2=1)
    print("Image with Edges:",type(edges), edges.shape)
    
    # Step 3: Performs Line estimation using cv2.HoughLinesP()
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)
    print("Image with Lines:",type(lines), lines.shape)  
    
    lines=np.squeeze(lines)
    
    _, image_cells = plt.subplots(1, 2, figsize=(10, 10))
    image_cells[0].imshow(lane_mask)
    image_cells[0].set_title('Lane Mask')
    image_cells[1].imshow(edges)
    image_cells[1].set_title('Edges')
    plt.show()

    return lines


lane_lines = estimate_lane_lines(segmentation)


_, image_cells = plt.subplots(1, 2, figsize=(10, 10))
image_cells[0].imshow(ground_mask)
image_cells[0].set_title('Ground Mask')
image_cells[1].imshow(dataset_handler.vis_lanes(lane_lines))
image_cells[1].set_title('Lane Lines')
plt.show()




print('\nEstimated 3D Drivable Space:')
dataset_handler.plot_free_space(ground_mask)



def merge_lane_lines(lines):
    """
    Merges lane lines to output a single line per lane, using the slope and intercept as similarity measures.
    Also, filters horizontal lane lines based on a minimum slope threshold.

    Arguments:
    lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.

    Returns:
    merged_lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.
    """
    
    # Step 0: Defines thresholds.
    slope_similarity_threshold = 0.1
    intercept_similarity_threshold = 40
    min_slope_threshold = 0.3
    
    
    # Step 1: Gets slope and intercept of lines.
    slopes, intercepts = get_slope_intecept(lines)
    
    # Step 2: Determines lines with slope less than horizontal slope threshold.
    slopes_filtered=slopes[np.abs(slopes)>min_slope_threshold]

    intercepts_filtered = intercepts[np.abs(slopes)>min_slope_threshold]
    lines_filtered=lines[np.abs(slopes)>min_slope_threshold]
    

    # Step 3: Iterates over all remaining slopes and intercepts and 
    # cluster lines that are close to each other using a slope and intercept threshold.
    N=lines_filtered.shape[0]
    clustered_bool=np.zeros((N))

    clustering_slope=slopes_filtered[0]
    clustering_intercept=intercepts_filtered[0]


    clustered_count=0
    cluster=[]
    cluster_dict=dict()
    cluster_size=0

    
    for j in range (0, len(lines_filtered)):
        if (clustered_bool[j]==0):
            clustering_slope=slopes_filtered[j]
            clustering_intercept=intercepts_filtered[j]
            k=0
            for i in range (0, len(lines_filtered)):
                if((np.abs(clustering_slope - slopes_filtered[i])<slope_similarity_threshold) and (np.abs(clustering_intercept-intercepts_filtered[i])< intercept_similarity_threshold)):
                    cluster.append([lines_filtered[i]])
                    clustered_count+=1
                    clustered_bool[i]=1
        
        if(len(cluster) != 0):
            cluster_dict.setdefault(cluster_size, []).append(cluster)
            cluster_size+=1
        if(clustered_count==len(lines_filtered)):
            break
        
        cluster=[]

    cluster_size=len(cluster_dict)
    
    
    # Step 4: Merges all lines in clusters using mean averaging
    cluster_mean_xy=dict()
    line_x1=0
    line_x2=0
    line_y1=0
    line_y2=0
    count_points=0

    merged_lines=np.empty([cluster_size,4])

    for i in range (0, cluster_size):
        an_array = np.array(cluster_dict[i]).reshape(-1)
        for j in range (0, len(an_array)):
            if((j%4)==0):
                line_x1+=an_array[j]
                count_points+=1
            if(j%4==1):
                line_y1+=an_array[j]
            if(j%4==2):
                line_x2+=an_array[j]
            if(j%4==3):
                line_y2+=an_array[j]
        line_x1=line_x1/count_points
        line_y1=line_y1/count_points
        line_x2=line_x2/count_points
        line_y2=line_y2/count_points

        merged_lines[i,:]=[line_x1, line_y1, line_x2, line_y2]

        count_points=0
        line_x1=0
        line_x2=0
        line_y1=0
        line_y2=0

    print('Merged Lines:\n', merged_lines, merged_lines.shape)

    return merged_lines


merged_lane_lines = merge_lane_lines(lane_lines)


# Extrapolates the lanes to start at the beginning of the road, and end at the end of the road.
max_y = dataset_handler.image.shape[0]
min_y = np.min(np.argwhere(road_mask == 1)[:, 0])

extrapolated_lanes = extrapolate_lines(merged_lane_lines, max_y, min_y)
final_lanes = find_closest_lines(extrapolated_lanes, dataset_handler.lane_midpoint)




_, image_cells = plt.subplots(1, 2, figsize=(10, 10))
image_cells[0].imshow(dataset_handler.vis_lanes(merged_lane_lines))
image_cells[0].set_title('Merged Lane Lines')
image_cells[1].imshow(dataset_handler.vis_lanes(final_lanes))
image_cells[1].set_title('Final Lanes')
plt.show()




### III - Computing Minimum Distance To Impact Using The Output of 2D Object Detection.
detections = dataset_handler.object_detection

print("\nDetection comes from Neural Net:")
print(detections)


# Filtering Out Unreliable Detections
def filter_detections_by_segmentation(detections, segmentation_output):
    """
    Filters 2D detection output based on a semantic segmentation map.

    Arguments:
    detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].
    
    segmentation_output -- tensor of dimension (HxW) containing pixel category labels.
    
    Returns:
    filtered_detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].

    """
    
    # Sets ratio threshold:
    ratio_threshold = 0.3  # If 1/3 of the total pixels belong to the target category, the detection is correct.
    
    area=0.0
    filtered_detections=[]
    
    for detection in detections:
        # Step 1: Computes number of pixels (area) belonging to the category for every detection.   detection[3]=x_max    detection[1]=x_min...
        area=(((np.asfarray(detection[3]))-(np.asfarray(detection[1])))*((np.asfarray(detection[4]))-(np.asfarray(detection[2]))))

        # Step 2: Devides the computed number of pixels by the area of the bounding box (total number of pixels).
        count_w=0
        count_p=0
        
        # Handles Car (index:10) and Pedestrians (index:4)
        for i in range (int(float(detection[2])), int(float(detection[4])) ):            
            for j in range (int(float(detection[1])), int(float(detection[3]))):
                
                # if pixel is in the Car segmentation.
                if(segmentation_output[i,j]==10):
                    count_w+=1
                    
                # If pixel is in the Pedestrians segmentation.
                if(segmentation_output[i,j]==4):
                    count_p+=1
        
        if(detection[0]=="Car"):
            ratio=count_w/area
        else:
            ratio=count_p/area
            
        # Step 3: If the ratio is greater than a threshold keep the detection. Else, remove the detection from the list of detections.
        if (ratio>ratio_threshold):
            filtered_detections.append(detection)
    print("\nFiltered Detections by Segmentation Map:")
    print(filtered_detections)

    return filtered_detections



filtered_detections = filter_detections_by_segmentation(detections, segmentation)

_, image_cells = plt.subplots(1, 2, figsize=(10, 10))
image_cells[0].imshow(dataset_handler.vis_object_detection(detections))
image_cells[0].set_title('Detections from Neural Net')
image_cells[1].imshow(dataset_handler.vis_object_detection(filtered_detections))
image_cells[1].set_title('Filtered Detections')
plt.show()


# Estimating Minimum Distance To Impact
def find_min_distance_to_detection(detections, x, y, z):
    """
    Filters 2D detection output based on a semantic segmentation map.

    Arguments:
    detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].
    
    x -- tensor of dimension (H, W) containing the x coordinates of every pixel in the camera coordinate frame.
    y -- tensor of dimension (H, W) containing the y coordinates of every pixel in the camera coordinate frame.
    z -- tensor of dimensions (H,W) containing the z coordinates of every pixel in the camera coordinate frame.
    
    Returns:
    min_distances -- tensor of dimension (N, 1) containing distance to impact with every object in the scene.

    """
    
    dist=0
    min_distances_l=[]
    
    for detection in detections:
        
        # Step 1: Computes distance of every pixel in the detection bounds.
        distances=[]
        for i in range (int(float(detection[2])), int(float(detection[4])) ):            
            for j in range (int(float(detection[1])), int(float(detection[3]))):
                x_p=x[i][j]
                y_p=y[i][j]
                z_p=z[i][j]
                dist=np.sqrt(x_p*x_p + y_p*y_p + z_p*z_p)
                distances.append(dist)
                dist=0

        # Step 2: Finds minimum distance.
        min_distances_l.append(min(distances))

    min_distances=np.array(min_distances_l)
    return min_distances


min_distances = find_min_distance_to_detection(filtered_detections, x, y, z)

print('\nMinimum distance to impact is: ' + str(min_distances))


font = {'family': 'serif','color': 'red','weight': 'normal','size': 12}

im_out = dataset_handler.vis_object_detection(filtered_detections)

for detection, min_distance in zip(filtered_detections, min_distances):
    bounding_box = np.asfarray(detection[1:5])
    plt.text(bounding_box[0], bounding_box[1] - 20, 'Distance to Impact:' + str(np.round(min_distance, 2)) + ' m', fontdict=font)

plt.imshow(im_out)


"""
print("\n \n \nFor Image 1\n")

dataset_handler = DatasetHandler()
dataset_handler.set_frame(1)
segmentation = dataset_handler.segmentation
detections = dataset_handler.object_detection
z = dataset_handler.depth

# Part 1
k = dataset_handler.k
x, y = xy_from_depth(z, k)
road_mask = np.zeros(segmentation.shape)
road_mask[segmentation == 7] = 1
x_ground = x[road_mask == 1]
y_ground = y[road_mask == 1]
z_ground = dataset_handler.depth[road_mask == 1]
xyz_ground = np.stack((x_ground, y_ground, z_ground))
p_final = ransac_plane_fit(xyz_ground)

# Part II
lane_lines = estimate_lane_lines(segmentation)
merged_lane_lines = merge_lane_lines(lane_lines)
max_y = dataset_handler.image.shape[0]
min_y = np.min(np.argwhere(road_mask == 1)[:, 0])

extrapolated_lanes = extrapolate_lines(merged_lane_lines, max_y, min_y)
final_lanes = find_closest_lines(extrapolated_lanes, dataset_handler.lane_midpoint)

# Part III
filtered_detections = filter_detections_by_segmentation(detections, segmentation)
min_distances = find_min_distance_to_detection(filtered_detections, x, y, z)

# Print Submission Info

final_lane_printed = [list(np.round(lane)) for lane in final_lanes]
print("###################################################")
print("\n SUBMITTED RESULTS")
print('plane:') 
print(list(np.round(p_final, 2)))
print('\n lanes:')
print(final_lane_printed)
print('\n min_distance')
print(list(np.round(min_distances, 2)))


# Original Image
plt.imshow(dataset_handler.image)

# Part I
dist = np.abs(dist_to_plane(p_final, x, y, z))

ground_mask = np.zeros(dist.shape)

ground_mask[dist < 0.1] = 1
ground_mask[dist > 0.1] = 0

plt.imshow(ground_mask)


# Part II
plt.imshow(dataset_handler.vis_lanes(final_lanes))


# Part III
font = {'family': 'serif','color': 'red','weight': 'normal','size': 9}

im_out = dataset_handler.vis_object_detection(filtered_detections)

for detection, min_distance in zip(filtered_detections, min_distances):
    bounding_box = np.asfarray(detection[1:5])
    plt.text(bounding_box[0], bounding_box[1] - 20, 'Distance to Impact:' + str(np.round(min_distance, 2)) + ' m', fontdict=font)

plt.imshow(im_out)
"""



