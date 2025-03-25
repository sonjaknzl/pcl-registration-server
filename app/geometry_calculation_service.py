import open3d as o3d
import numpy as np

from app.algorithms.fgr_service import FGRService
from app.algorithms.ransac_icp_service import RansacICPService
from app.algorithms.teaserpp_service import TeaserPP

class GeometryCalculationService:
    
    def __init__(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud):
        self.source = source
        self.target = target

    def calculate_transformation_matrix_with_algorithm(self, algorithm: int) -> np.ndarray:
        ## SERVICE -> SWITCH CASE ALGORITHM INTERFACE
        match algorithm:
            case 0:
                self.algorithm = RansacICPService()
            case 1:
                self.algorithm = FGRService()
            case 2:
                self.algorithm = TeaserPP()
            case _:
                raise ValueError(f"Invalid algorithm value: {algorithm}")
        
        return self.algorithm.calculate_transformation_matrix(self.source, self.target)