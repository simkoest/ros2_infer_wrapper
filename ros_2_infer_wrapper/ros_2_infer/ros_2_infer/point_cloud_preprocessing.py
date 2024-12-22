import numpy as np

def point_cloud_create_bev_map(data, boundaries, bev_size):
    minX = boundaries['minX']
    maxX = boundaries['maxX']
    minY = boundaries['minY']
    maxY = boundaries['maxY']
    minZ = boundaries['minZ']
    maxZ = boundaries['maxZ']

    
    mask = np.where((data[:, 0] >= minX) & (data[:, 0] <= maxX) &
                    (data[:, 1] >= minY) & (data[:, 1] <= maxY) &
                    (data[:, 2] >= minZ) & (data[:, 2] <= maxZ))
    data = data[mask]
    data[:, 2] = data[:, 2] - minZ

    bev_height, bev_width = bev_size
    discretization = (boundaries["maxX"] - boundaries["minX"]) / bev_height

    Height = 608 + 1
    Width = 608 + 1
    PointCloud = np.copy(data)
    
    #PointCloud = lidarData.reshape(-1, 4)        
    
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / discretization))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / discretization) + Width / 2)

    # sort-3times
    sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[sorted_indices]
    #test = PointCloud[:,0:2]

    _, unique_indices, unique_counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[unique_indices]

    # Height Map, Intensity Map & Density Map
    heightMap = np.zeros((Height, Width))
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(boundaries['maxZ'] - boundaries['minZ']))
    heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height

    normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((3, Height - 1, Width - 1))
    RGB_Map[2, :, :] = densityMap[:bev_height,:bev_width]  # r_map
    RGB_Map[1, :, :] = heightMap[:bev_height,:bev_width]  # g_map
    RGB_Map[0, :, :] = intensityMap[:bev_height,:bev_width]  # b_map
    
    return RGB_Map    
    #bev_map = torch.from_numpy(RGB_Map)
    #bev_map = bev_map[np.newaxis, ...]
    #bev_map = bev_map.cpu().numpy().astype(np.float32)


def point_cloud_normalize(data):
    centroid = np.mean(data, axis=0)
    data -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(data)**2,axis=-1)))
    data /= furthest_distance
    return data

