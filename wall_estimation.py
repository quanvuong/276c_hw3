# Estimating the plane given a pointcloud

import open3d as o3d
import numpy as np
from plane_estimation_table import get_distance_to_plane, \
    get_pcd, get_plane

np.random.seed(0)

xs = np.loadtxt('hallway1b.asc')

# Since RANSAC only works reliable if
# the majority of the points being consider belong to the plane.
# We remove points too far away from the table.
# xs = xs[xs[:, 2] < 2.1]
# xs = xs[xs[:, 2] > 0.5]

# xs = xs[xs[:, 0] < 0.5]
# xs = xs[xs[:, 0] > -0.5]

# xs = xs[xs[:, 1] < 0.25]

print('Number of point after filtering', xs.shape[0])

num_points_for_plane_estimation = 3
inlier_threshold = 0.15
inlier_percentage_threshold = 0.3

# We follow the RANSAC algorithm
# given here.
# http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FISHER/RANSAC/
for i in range(300):

    print()
    print('iteration', i)

    # Select a small number of points to estimate plane
    indexes = np.random.choice(xs.shape[0],
                               size=num_points_for_plane_estimation)
    selected_xs = xs[indexes]

    plane_centroid, plane_normal = get_plane(selected_xs)

    distance = get_distance_to_plane(xs,
                                     plane_centroid,
                                     plane_normal)

    distance = np.abs(distance)

    inliers = xs[distance < inlier_threshold]
    outliers = xs[distance >= inlier_threshold]

    print('num inliers', len(inliers))
    print('num outlier', len(outliers))
    print('threshold', inlier_percentage_threshold * xs.shape[0])

    if len(inliers) > inlier_percentage_threshold * xs.shape[0]:

        print('plane centroid', plane_centroid)
        print('plane_normal', plane_normal)

        # Red
        inlier_pcd = get_pcd(inliers, [1., 0., 0.])

        # Green
        outlier_pcd = get_pcd(outliers, [0., 1., 0.])

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0, 0, 0])

        o3d.visualization.draw_geometries(
            [inlier_pcd, outlier_pcd, mesh_frame])
        # break

        xs = outliers
