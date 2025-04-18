import open3d as o3d
import numpy as np

from app.algorithms.fgr_service import FGRService
from app.algorithms.fgr_icp_service import FGRICPService
from app.algorithms.ransac_icp_service import RansacICPService
from app.algorithms.ransac_service import RansacService
from app.algorithms.teaserpp_service import TeaserPPService
from app.algorithms.teaserpp_icp_service import TeaserPPICPService

class GeometryCalculationService:
    
    def __init__(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud):
        self.source = source
        self.target = target

    def calculate_transformation_matrix_with_algorithm(self, algorithm: int) -> np.ndarray:
        ## SERVICE -> SWITCH CASE ALGORITHM INTERFACE
        match algorithm:
            case 0:
                self.algorithm = RansacService()
            case 1:
                self.algorithm = RansacICPService()
            case 2:
                self.algorithm = FGRService()
            case 3:
                self.algorithm = FGRICPService()
            case 4:
                self.algorithm = TeaserPPService()
            case 5:
                self.algorithm = TeaserPPICPService()
            case _:
                raise ValueError(f"Invalid algorithm value: {algorithm}")
        
        return self.algorithm.calculate_transformation_matrix(self.source, self.target)