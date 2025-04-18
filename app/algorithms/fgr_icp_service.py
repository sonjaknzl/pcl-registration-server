import copy
import time
import open3d as o3d
import numpy as np
from  ..model.i_algorithm_service import IAlgorithmService

class FGRICPService(IAlgorithmService):

    def calculate_transformation_matrix(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud):
        voxel_size = 50 # 1 = 1MM FOR DATA SET
        
        # PREPARE DATA SET
        source, target, source_down, target_down, source_fpfh, target_fpfh = self.prepare_dataset(source, target, voxel_size)
        
        # FGR
        result_fast = self.execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        start = time.time()
        print("Fast global registration took %.3f sec.\n" % (time.time() - start))
        print(result_fast.transformation)
        
        # ICP
        result_icp = self.refine_registration(source, target, source_fpfh, target_fpfh, result_fast, voxel_size)
        print(result_icp.transformation)
        
        self.draw_registration_result(source, target, result_icp.transformation)
        return result_icp
    
    def execute_fast_global_registration(self, source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        distance_threshold = voxel_size * 3
        print(":: Apply fast global registration with distance threshold %.3f" \
                % distance_threshold)
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        
        return result
    
    
    