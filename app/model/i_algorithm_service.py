from abc import ABC, abstractmethod
import open3d as o3d
import numpy as np
import copy

class IAlgorithmService(ABC):
    @abstractmethod
    def calculate_transformation_matrix(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud) -> np.ndarray:
        pass
    
    def prepare_dataset(self, source, target, voxel_size):
        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)
        return source, target, source_down, target_down, source_fpfh, target_fpfh
    
    def preprocess_point_cloud(self, pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 200
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 500
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh
    
    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)

        # Assign uniform colors to the source and target
        source_temp.paint_uniform_color([1, 0, 0])
        target_temp.paint_uniform_color([0, 1, 1])

        # Apply the transformation to the source point cloud
        source_temp.transform(transformation)

        # Combine both point clouds into one
        combined_pcd = source_temp + target_temp

        # Save the combined point clouds into one file
        o3d.io.write_point_cloud("combined_point_cloud.ply", combined_pcd)
        print("Point clouds saved to 'combined_point_cloud.ply'.")
        
    def refine_registration(self, source, target, source_fpfh, target_fpfh, preliminary_result, voxel_size):
        distance_threshold = voxel_size * 0.4
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)
        
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=30)
        )
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=30)
        )
        
        # Perform ICP registration with point-to-plane estimation
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, preliminary_result.transformation, 
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        return result
