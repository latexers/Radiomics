import nibabel as nib
import numpy as np
import utils
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.segmentation import slic
import os

def slic_supervoxel(image_path, mask_path, n_clusters=100, smooth_sigma=1.0, output_path=None):
    """
    使用 K-means 聚类进行超体素分割。结合 SimpleITK 和 KMeans 聚类对 3D 图像进行处理，
    仅对 mask 为 True 的区域进行聚类，并将结果保存为 NIfTI 文件。

    Args:
        image_path (str): 输入 3D 图像的路径，支持 NIfTI 格式 (如 .nii.gz)。
        mask (numpy.ndarray or None): 二值掩码图像，指定只对掩码区域进行聚类，默认 None 表示对整个图像进行处理。
        n_clusters (int): 超体素的数量（即聚类数量），默认 100。
        smooth_sigma (float): 高斯平滑的标准差，默认 1.0。
        output_path (str or None): 输出文件的路径，保存为 .nii.gz 格式。如果为 None，则不保存。

    Returns:
        np.ndarray: 超体素标签图像。
    """

    image = sitk.ReadImage(image_path)
    smoothed_image = sitk.SmoothingRecursiveGaussian(image, sigma=smooth_sigma)
    image_array = sitk.GetArrayFromImage(smoothed_image)

    if mask_path is not None:
        mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask)
        image_array = image_array * mask_array

    z, y, x = np.indices(image_array.shape)
    pixel_values = image_array.flatten()
    spatial_coordinates = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
    features = np.concatenate([spatial_coordinates, pixel_values[:, np.newaxis]], axis=-1)

    if mask is not None:
        features = features[mask_array.flatten() > 0]

    # K-means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(features)
    labels = kmeans.labels_
    segmented_image = np.zeros(image_array.shape, dtype=np.int32)

    if mask is not None:
        segmented_image[mask_array > 0] = labels + 1  # 超体素标签从 1 开始
    else:
        segmented_image = labels.reshape(image_array.shape)

    if output_path is not None:
        segmented_image_itk = sitk.GetImageFromArray(segmented_image)
        segmented_image_itk.CopyInformation(image)
        sitk.WriteImage(segmented_image_itk, output_path)
        print(f"Supervoxel segmentation saved to {output_path}")

    return segmented_image


def cluster_image_with_mask(image_path, mask_path, num_classes, output_path=None):
    """
    使用聚类算法对图像中的掩码区域进行聚类。

    Args:
        image (numpy.ndarray): 输入的图像数组。
        mask (numpy.ndarray): 输入的掩码数组。
        num_classes (int): 指定的聚类类别数。

    Returns:
        numpy.ndarray: 聚类后的图像。
    """
    image = sitk.ReadImage(image_path)
    img_array = sitk.GetArrayFromImage(image)
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)
    masked_image = img_array[mask_array > 0] 
    masked_image_reshaped = masked_image.reshape(-1, 1)

    kmeans = KMeans(n_clusters=num_classes, random_state=42)
    kmeans.fit(masked_image_reshaped)


    cluster_labels = kmeans.labels_

    # 创建聚类后的图像
    clustered_image = np.zeros_like(img_array)

    clustered_image[mask_array > 0] = cluster_labels + 1 
    if output_path is not None:

        segmented_image_itk = sitk.GetImageFromArray(clustered_image)
        segmented_image_itk.CopyInformation(image) 

        sitk.WriteImage(segmented_image_itk, output_path)
        print(f"Habitat segmentation saved to {output_path}")
    return clustered_image



def segment_3d_image_with_slic(image_path, mask_path, num_segments, compactness=10, output_path=None):
    """
    使用SLIC算法对3D图像进行超像素分割。
    
    Args:
        image_path (str): 输入的3D图像路径。
        mask_path (str): 输入的掩码图像路径。
        num_segments (int): 超像素的数量。
        compactness (float): 控制超像素分割时空间和颜色的平衡。
        output_path (str, optional): 分割结果保存路径（NIfTI格式）。
        
    Returns:
        numpy.ndarray: 3D图像的超像素标签。
    """
    # 加载图像和掩码
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        raise FileNotFoundError("输入的图像或掩码路径无效")
    
    # 读取图像和掩码
    image = sitk.ReadImage(image_path)
    img_array = sitk.GetArrayFromImage(image)  
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)
    mask_array = np.where(mask_array > 0, 1, 0)

    segmented_image_3d = np.zeros_like(img_array)
    for i in range(img_array.shape[2]):
        segmented_image_3d[:,:,i] = slic(img_array[:,:,i], n_segments=num_segments, compactness=compactness,channel_axis=None)
    if output_path is not None:
        segmented_image_3d = segmented_image_3d.astype(np.uint8)*mask_array
        segmented_image_itk = sitk.GetImageFromArray(segmented_image_3d.astype(np.uint8))
        segmented_image_itk.CopyInformation(image)  # 保留原始图像的空间信息
        sitk.WriteImage(segmented_image_itk, output_path)
        print(f"Segmentation saved to {output_path}")
    
    return segmented_image_3d


if __name__ == '__main__':
    image_path = r'F:\\BaiduNetdiskDownload\\case_00206\\imaging.nii.gz'  # 替换为你的图像路径
    mask_path = r'F:\\BaiduNetdiskDownload\\case_00206\\segmentation.nii.gz'  # 替换为你的掩码路径
    supervoxel_hibatat = slic_supervoxel(image_path, mask_path, n_clusters=10, smooth_sigma=1.0, output_path="supervoxel_result.nii.gz")
    slic_habitat = segment_3d_image_with_slic(image_path, mask_path, 10, output_path="slic_result.nii.gz")
    kmeans_habitat = cluster_image_with_mask(image_path, mask_path, 10, output_path="kmeans_result.nii.gz")