import cupy as cp
import cv2

def Mat2Tensor(gpu_mat: cv2.cuda.GpuMat, device: str = 'cuda:0') -> cp.ndarray:
    """
    Convert a GpuMat to a CudaTensor.
    Args:
        gpu_mat: GpuMat to convert
        device: device to store the CudaTensor

    Returns: CudaTensor converted from GpuMat

    """
def Tensor2Mat(tensor):
    """
    Convert a CudaTensor to a GpuMat.
    Args:
        tensor: CudaTensor to convert
    Returns: GpuMat converted from CudaTensor
    """
