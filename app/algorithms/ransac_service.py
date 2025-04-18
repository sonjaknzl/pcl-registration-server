import copy
import open3d as o3d
import numpy as np
from  ..model.i_algorithm_service import IAlgorithmService

class RansacService(IAlgorithmService):

    def calculate_transformation_matrix(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud):
        voxel_size = 50 # = 1CM FOR DATA SET
        
        # PREPARE DATA SET
        source, target, source_down, target_down, source_fpfh, target_fpfh = self.prepare_dataset(source, target, voxel_size)
        
        # RANSAC
        result_ransac = self.execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        print("RANSAC Transformation Matrix:")
        print(result_ransac.transformation)  
        
        self.draw_registration_result(source, target, result_ransac.transformation)
        
        return result_ransac
    
    def execute_global_registration(self, source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        distance_threshold = voxel_size * 3
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result
    