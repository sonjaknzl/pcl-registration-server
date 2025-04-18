import numpy as np
import open3d as o3d
from app.model.i_algorithm_service import IAlgorithmService
import teaserpp_python
from .helpers import *

class TeaserPPICPService(IAlgorithmService):

    def calculate_transformation_matrix(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud):
       
        voxel_size = 50 # = 1M FOR DATA SET
       
        src_down: o3d.geometry.PointCloud = source.voxel_down_sample(voxel_size)
        tgt_down: o3d.geometry.PointCloud = target.voxel_down_sample(voxel_size)
        
        src_down_xyz = pcd2xyz(src_down) 
        tgt_down_xyz = pcd2xyz(tgt_down)
        
        src_feats = extract_fpfh(src_down,voxel_size)
        tgt_feats = extract_fpfh(tgt_down,voxel_size)
        
        corrs_A, corrs_B = find_correspondences(
        src_feats, tgt_feats, mutual_filter=True)
        A_corr = src_down_xyz[:,corrs_A] # np array of size 3 by num_corrs
        B_corr = tgt_down_xyz[:,corrs_B] # np array of size 3 by num_corrs
        
        num_corrs = A_corr.shape[1]
        print(f'FPFH generates {num_corrs} putative correspondences.')
        
        NOISE_BOUND = voxel_size
        teaser_solver = get_teaser_solver(NOISE_BOUND)
        teaser_solver.solve(A_corr,B_corr)
        solution = teaser_solver.getSolution()
        scale = solution.scale
        rotation = solution.rotation
        translation = solution.translation
        # T_teaser = Rt2T(R_teaser,t_teaser)
                    
        
        
        
    #    # Convert Open3D point clouds to numpy arrays
    #     src_points = np.asarray(source.points).T  
    #     tgt_points = np.asarray(target.points).T  
        
    #     # Set up TEASER++ solver parameters
    #     solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    #     solver_params.cbar2 = 1
    #     solver_params.noise_bound = 0.1  # Adjust as needed based on noise level
    #     solver_params.estimate_scaling = True
    #     solver_params.rotation_estimation_algorithm = (
    #         teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    #     )
    #     solver_params.rotation_gnc_factor = 1.4
    #     solver_params.rotation_max_iterations = 100
    #     solver_params.rotation_cost_threshold = 1e-12
        
    #     print("Source points shape:", src_points.shape)
    #     print("Target points shape:", tgt_points.shape)

    #     print("Solver parameters initialized:", solver_params)

    #     # Initialize TEASER++ solver
    #     solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    #     # Solve the transformation between source and target point clouds
    #     solver.solve(src_points, tgt_points)

    #     # Retrieve the solution
    #     solution = solver.getSolution()
    #     scale = solution.scale
    #     rotation = solution.rotation
    #     translation = solution.translation

        print("Transformation estimation completed.")
        print(f"Scale: {scale}")
        print(f"Rotation:\n{rotation}")
        print(f"Translation: {translation}")

    #     # Output the inliers (optional, for debugging or evaluation)
    #     scale_inliers = solver.getScaleInliers()
    #     translation_inliers = solver.getTranslationInliers()


        # Construct the transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation * scale
        transformation_matrix[:3, 3] = translation
    

        registration_result = o3d.pipelines.registration.RegistrationResult()
        registration_result.transformation = transformation_matrix
        
        #ICP
        result_icp = self.refine_registration(source, target, src_feats, tgt_feats, registration_result, voxel_size)
        print(result_icp.transformation)
        
        
        self.draw_registration_result(source, target, result_icp.transformation)
        return result_icp