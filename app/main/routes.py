import copy
from app.main import bp
from flask import Flask, request
import open3d as o3d
import numpy as np
import json
from app.geometry_calculation_service import GeometryCalculationService
import os
import time

@bp.route("/", methods=["POST"])
def calculate_location():
    # DATA CLASS FROM JSON
    data = request.get_json()
    
    # SET ALGORITHM & TARGET
    algorithm = int(data.get('algorithm', -1))
    targetmodel = data.get('targetmodel', -1)
    sourcepoints = int(data.get('sourcemodel', -1))
    
    # Define the folder paths for source and target point clouds
    assets_folder = os.path.join(
        os.path.dirname(__file__),  
        '..',                       
        'assets'                    
    )
    
    target_folder = os.path.join(assets_folder, 'target_pointclouds')
    
    target_folder = os.path.abspath(target_folder)
    
    # Construct file paths dynamically based on targetmodel and sourcemodel
    target_model_file = os.path.join(target_folder, f"{targetmodel}")
    
    # Load the target point cloud
    if not os.path.isfile(target_model_file):
        raise FileNotFoundError(f"Point cloud file not found: {target_model_file}")
    
    target_point_cloud = o3d.io.read_point_cloud(target_model_file)
    if not target_point_cloud.has_points():
        raise ValueError(f"Failed to load point cloud from {target_model_file}")
    
    # FORMAT DATA TO POINTCLOUD
    sourcepoints = np.array(data['sourcepoints'])
    source_point_cloud = o3d.geometry.PointCloud()
    source_point_cloud.points = o3d.utility.Vector3dVector(sourcepoints)
    
    draw(source_point_cloud, target_point_cloud)
    
    # GEOMETRY CALCULATION SERVICE
    geometry_service = GeometryCalculationService(source_point_cloud, target_point_cloud)
    
    start_time = time.perf_counter()
    transformation_matrix = geometry_service.calculate_transformation_matrix_with_algorithm(algorithm)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    transformation_matrix_list = transformation_matrix.transformation.tolist()
    
    # RESPONSE
    response = {
    "transformation_matrix": transformation_matrix_list,
    "elapsed_time": elapsed_time
    }
    return json.dumps(response), 201

@bp.route("/", methods=["GET"])
def test():
    return "OK"

    
def draw_source(source):
    # Create deep copies of the source and target point clouds
    source_temp = copy.deepcopy(source)
    
    # Assign uniform colors to the source and target
    source_temp.paint_uniform_color([1, 0, 0])  # Red for the source point cloud
    
    # Combine both point clouds into one
    pre_pcd = source_temp
    
    # Save the combined point clouds into one file
    o3d.io.write_point_cloud("pre_point_cloud.ply", pre_pcd)
    print("Point clouds saved to 'combined_point_cloud.ply'.")

def draw(source, target):
    # Create deep copies of the source and target point clouds
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    # Assign uniform colors to the source and target
    source_temp.paint_uniform_color([1, 0, 0]) 
    target_temp.paint_uniform_color([0, 1, 1])  
    
    # Combine both point clouds into one
    precombined_pcd = source_temp + target_temp
    
    # Save the combined point clouds into one filev
    o3d.io.write_point_cloud("precombined_point_cloud_source.ply", source_temp)
    o3d.io.write_point_cloud("precombined_point_cloud.ply", precombined_pcd)
    print("Point clouds saved to 'combined_point_cloud.ply'.")
    