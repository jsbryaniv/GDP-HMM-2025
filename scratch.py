
# Import libraries
import os
import json
import datasets
from huggingface_hub import login, snapshot_download

# Set up hugging face
with open("config.json", "r") as f:
    config = json.load(f)
    hf_token = config["HUGGINGFACE_TOKEN"]
login(token=hf_token)

# Download the dataset
snapshot_download(
    repo_id="Jungle15/Radiotherapy_HaN_Lung_AIRTP", 
    repo_type="dataset", 
    local_dir="data/dicoms/"
)

# Done
print('Done')




# # Import libraries
# import os
# import time
# import torch
# import pydicom
# import numpy as np
# import SimpleITK as sitk
# import torch.nn.functional as F
# from scipy import ndimage
# from skimage.draw import polygon


# # -------------------------------------------------------------
# # ------------------------- Constants -------------------------
# # -------------------------------------------------------------

# # Define the OAR priorities
# OAR_PRIORITIES = {
#     ## High Priority ##
#     'brain':               0,
#     'brain_stem':          0,
#     'cord':                0,
#     'optic_nrv_l':         0,
#     'optic_nrv_r':         0,
#     'eye_l':               0,
#     'eye_r':               0,
#     'cochlea_l':           0,
#     'cochlea_r':           0,
#     'larynx':              0,
#     'brachial_plex_l':     0,
#     'brachial_plex_r':     0,
#     'esophagus':           0,
#     'esophagus_cerv':      0,
#     ## Medium Priority ##
#     'mandible':            1,
#     'oral_cavity':         1,
#     'oral_cavity-ptv':     1,
#     'parotid_l':           1,
#     'parotid_r':           1,
#     'parotid_l_prv':       1,
#     'parotid_r_prv':       1,
#     'parotid_total':       1,
#     'constrictors_p':      1,
#     'constrictor prv':     1,
#     'sub_mandib_l':        1,
#     'sub_mandib_r':        1,
#     'crico_p_inlet':       1,
#     'pituitary':           1,
#     'nasal_cavity':        1,
#     ## Low Priority ##
#     'body-ptv':            2,
#     'FD_artifact':         2,
#     'lung_l':              2,
#     'lung_r':              2,
#     'lung_total':          2,
#     'mastoid_l':           2,
#     'mastoid_r':           2,
#     'ext_aud_canal_l':     2,
#     'ext_aud_canal_r':     2,
#     'semi_cir_canal_l':    2,
#     'semi_cir_canal_r':    2,
#     'optic_nrv_prv_l':     2,
#     'optic_nrv_prv_r':     2,
#     'cord_prv':            2,
#     'brain_stem_prv':      2,
#     'larynx-ptv':          2,
#     'eval_carotid_art':    2,
#     'thyroid':             2,
#     'lips':                2,
#     'zRing':               2,
#     'zRing2':              2,
#     'CouchInterior':       2,
#     'CouchSurface':        2,
#     'zPtvLowOpti1':        2,
# }


# # ----------------------------------------------------------------
# # ------------------------- Data Loading -------------------------
# # ----------------------------------------------------------------

# def load_sitk(path):
#     """
#     This function loads a DICOM file using SimpleITK.

#     Arguments:
#         path (str): Path to the DICOM file.

#     Returns:
#         sitk_image (SimpleITK.Image): SimpleITK image object.
#     """

#     # Set up SimpleITK reader
#     reader = sitk.ImageSeriesReader()
#     dicom_files = reader.GetGDCMSeriesFileNames(path)
#     reader.SetFileNames(dicom_files)

#     # Read the image
#     sitk_image = reader.Execute()

#     # Return the image
#     return sitk_image

# def load_scan(path):
#     """
#     This function loads a CT scan from a DICOM directory.

#     Arguments:
#         path (str): Path to the DICOM directory.

#     Returns:
#         scan (numpy.ndarray): 3D numpy array of the CT scan.
#     """

#     # Load the image
#     ct_sitk = load_sitk(path)

#     # Get the array
#     scan = sitk.GetArrayFromImage(ct_sitk)

#     # Return the image array and SimpleITK image
#     return scan

# def load_rxdose(path_rp):
#     """
#     This function loads the prescription dose from a DICOM RT Plan file.

#     Arguments:
#         rp_path (str): Path to the DICOM RT Plan file.

#     Returns:
#         rxdoses (list): List of prescription doses.
#     """

#     # Read the RP DICOM file
#     ds = pydicom.dcmread(path_rp)

#     # Initialize dose list
#     rxdoses = []

#     # Loop over Dose References
#     for ref in ds.DoseReferenceSequence:

#         # Get dose
#         x = ref.get("TargetPrescriptionDose", None)
#         # If not available, try to get from DeliveryMaximumDose
#         if x is None:
#             x = ref.get("DeliveryMaximumDose", None)
#         # If not available, try to get from OrganAtRiskMaximumDose
#         if x is None:
#             x = ref.get("OrganAtRiskMaximumDose", None)
#         # If not available, skip this reference
#         if x is None:
#             continue

#         # Append to doses list
#         rxdoses.append(float(x))

#     # Sort doses
#     rxdoses = sorted(rxdoses)

#     # Return doses
#     return rxdoses

# def load_beaminfo(path_rp, path_ct):
#     """
#     This function loads the beam information from a DICOM RT Plan file.

#     Arguments:
#         path_rp (str): Path to the DICOM RT Plan file.
#         path_ct (str): Path to the DICOM CT file.

#     Returns:
#         beam_info (dict): Dictionary containing beam angles, isocenter position, and spacing.
#     """

#     # Load dicoms
#     rp = pydicom.dcmread(path_rp)
#     ct_sitk = load_sitk(path_ct)

#     # Get info from sitk
#     spacing = ct_sitk.GetSpacing()
#     origin = ct_sitk.GetOrigin()

#     # Get angles
#     angles = []
#     for beam_seq in rp.BeamSequence:
#         if not beam_seq.TreatmentDeliveryType == 'TREATMENT':
#             continue
#         for control_point in beam_seq.ControlPointSequence:
#             if 'GantryAngle' in control_point:
#                 angle = float(control_point.GantryAngle)
#                 angles.append(angle)
    
#     # Get isocenter
#     isocenter = [float(x) for x in beam_seq.ControlPointSequence[0].IsocenterPosition]
#     isocenter = [
#         int((isocenter[0] - origin[0]) / spacing[0]),
#         int((isocenter[1] - origin[1]) / spacing[1]),
#         int((isocenter[2] - origin[2]) / spacing[2]),
#     ]

#     # Create beam info dictionary
#     beam_info = {
#         'angles': angles,
#         'isocenter': isocenter,
#         'spacing': spacing,
#     }

#     # Return beam info
#     return beam_info

# def load_dose(path_rd, path_ct):
#     """
#     This function loads a dose from a DICOM file. It optionally matches the dose to a CT scan.

#     Arguments:
#         path_rd (str): Path to the DICOM RD file.
#         path_ct (str): Path to the DICOM CT file.

#     Returns:
#         dose_array (numpy.ndarray): 3D numpy array of the dose.
#     """

#     # Load dicom
#     rd = pydicom.dcmread(path_rd)
#     dose_grid = rd.pixel_array * rd.DoseGridScaling

#     # Convert to sitk
#     dose_sitk = sitk.GetImageFromArray(dose_grid)
#     spacing_z = rd.GridFrameOffsetVector[1] - rd.GridFrameOffsetVector[0] if len(rd.GridFrameOffsetVector) > 1 else 2.0
#     dose_sitk.SetSpacing((float(rd.PixelSpacing[0]), float(rd.PixelSpacing[1]), spacing_z))
#     dose_sitk.SetOrigin(tuple(rd.ImagePositionPatient))
#     dose_sitk.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

#     # Resample dose to match CT
#     match_ct = load_sitk(path_ct)
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetReferenceImage(match_ct)
#     resampler.SetInterpolator(sitk.sitkLinear)
#     resampler.SetDefaultPixelValue(0.0)
#     dose_resampled = resampler.Execute(dose_sitk)

#     # Extract array
#     dose_array = sitk.GetArrayFromImage(dose_resampled)
    
#     # Return the dose array
#     return dose_array

# def load_structures(path_rs, path_ct, names=None):
#     """
#     This function loads structure masks from a DICOM file. It optionally matches the structures to a CT scan.

#     Arguments:
#         path_rs (str): Path to the DICOM Structure file.
#         path_ct (str): Path to the DICOM CT file.
#         names (list): List of structure names to load. If None, loads all structures.

#     Returns:
#         masks (dict): Dictionary of structure masks, where keys are structure names and values are 3D numpy arrays.
#     """

#     # Load dicom
#     ds = pydicom.dcmread(path_rs)
#     structure_set = ds.StructureSetROISequence
#     roi_contours = ds.ROIContourSequence

#     # Get the image shape and spacing
#     match_ct = load_sitk(path_ct)
#     spacing = match_ct.GetSpacing()
#     origin = match_ct.GetOrigin()
#     shape_xyz = match_ct.GetSize()
#     shape_zyx = shape_xyz[::-1]

#     # Initialize masks
#     structures = {}

#     # Loop through each ROI
#     for roi, roi_contour in zip(structure_set, roi_contours):

#         # Get name of the ROI
#         name = roi.ROIName
#         if (names is not None) and (name not in names):
#             continue

#         # Initialize mask
#         mask = np.zeros(shape_zyx, dtype=bool)

#         # Loop through each contour sequence
#         for sequence in roi_contour.ContourSequence:

#             # Get contour coordinates
#             coords = np.array(sequence.ContourData).reshape(-1, 3)  # Reshape to (N, 3)
            
#             # Transform physical coordinates to voxel coordinates
#             # voxels = np.round((coords - origin) / spacing).astype(np.int32)
#             voxels = np.round((coords - origin) / spacing).astype(int)
#             voxels = np.clip(voxels, 0, np.array(shape_xyz) - 1)

#             # Loop over z slices
#             for z_slice in set(voxels[:, 2]):

#                 # Get points in the slice
#                 slice_points = np.array([(x, y) for x, y, z in voxels if z == z_slice])

#                 # Skip slices with less than 3 points
#                 if slice_points.shape[0] < 3:
#                     continue

#                 # Create mask
#                 rr, cc = polygon(slice_points[:, 0], slice_points[:, 1], shape_xyz[:2])
#                 mask[z_slice, cc, rr] = 1  # Mask coordinates are (z, y, x)

#         # Fill holes in the mask
#         mask = ndimage.binary_fill_holes(mask)

#         # Add to masks
#         structures[name] = mask
    
#     # Return masks
#     return structures

# def make_oars_array(structures, n_channels=None, mapping=None):
#     """
#     This function merges OAR structure masks into a single mask with multiple channels.
#     Each channel corresponds to a different priority level for the OARs.

#     Arguments:
#         structures (dict): Dictionary of structure masks, where keys are structure names and values are 3D numpy arrays.
#         n_channels (int): Number of channels in the output mask. If None, determined from mapping.
#         mapping (dict): Dictionary mapping structure names to channel indices. If None, uses default mapping.

#     Returns:
#         mask (numpy.ndarray): 3D numpy array of the merged OAR mask with shape (n_channels, Z, Y, X).
#     """

#     # Set defaults
#     if mapping is None:
#         mapping = OAR_PRIORITIES
#     if n_channels is None:
#         n_channels = max(mapping.values()) + 1    

#     # Initialize mask
#     ref_arr = structures[list(structures.keys())[0]]  # Use first structure as reference
#     mask = np.zeros((n_channels, *ref_arr.shape), dtype=bool)

#     # Loop over structures
#     for name, struct in structures.items():

#         # Get channel index
#         channel = mapping.get(name, None)
#         if channel is None:
#             continue

#         # Add to mask
#         mask[channel] = np.logical_or(mask[channel], struct)

#     # Return mask
#     return mask

# def make_ptvs_array(structures, rxdoses):
#     """
#     This function creates a PTV array from the structure masks and prescription doses.
#     It merges all the PTVs into a single array where each voxel is assigned the prescription dose.

#     Arguments:
#         structures (dict): Dictionary of structure masks, where keys are structure names and values are 3D numpy arrays.
#         rxdoses (list): List of prescription doses.

#     Returns:
#         ptvs (numpy.ndarray): 3D numpy array of the PTVs, where each voxel is assigned the prescription dose.
#     """

#     # Initialize ptvs
#     ref_arr = structures[list(structures.keys())[0]]  # Use first structure as reference
#     ptvs = np.zeros(ref_arr.shape, dtype=float)

#     # Find all the ptv keys and sort them by priority
#     def sort_order(x):
#         if 'high' in x.lower():
#             return 0
#         elif 'mid' in x.lower():
#             return 1
#         elif 'low' in x.lower():
#             return 2
#         else:
#             return 3
#     ptv_keys = [key for key in structures.keys() if 'ptv' in key.lower()]
#     ptv_keys.sort(key=sort_order)

#     # Sort rxdoses
#     rxdoses = sorted(rxdoses)[::-1]

#     # Loop over ptv keys
#     for i, key in enumerate(ptv_keys):
#         if i >= len(rxdoses):
#             break

#         # Get the ptv mask and dose
#         ptv_mask = structures[key] * (ptvs == 0)  # Avoid overwriting existing doses
#         dose = rxdoses[i]

#         # Fill in the ptvs
#         ptvs[ptv_mask] = dose

#     # Return the ptvs
#     return ptvs

# def make_beam_array(ptvs, beam_info, calculation_scale=2):
#     """
#     This function creates a beam array from the PTVs and beam information.

#     Arguments:
#         ptvs (numpy.ndarray): 3D numpy array of the PTVs, where each voxel is assigned the prescription dose.
#         beam_info (dict): Dictionary containing beam angles, isocenter position, and spacing.

#     Returns:
#         beam (numpy.ndarray): 3D numpy array of the beam, where each voxel is assigned the beam dose.
#     """

#     # Extract info
#     angles = beam_info['angles']
#     spacing = beam_info['spacing']
#     isocenter = beam_info['isocenter']

#     # Convert xyz to zyx
#     spacing = spacing[::-1]
#     isocenter = isocenter[::-1]

#     # Get the target
#     target = ptvs > 0

#     # Rescale target
#     if calculation_scale != 1:
#         spacing = [x * calculation_scale for x in spacing]
#         isocenter = [x // calculation_scale for x in isocenter]
#         reshape_size = [x // calculation_scale for x in target.shape]
#         target, transform_params = resize_image_3d(target[None, None, ...], reshape_size, return_params=True)
#         target = target[0, 0, ...]

#     # Get the beam plate
#     beam = get_allbeam_plate(target, isocenter, spacing, angles)

#     # Rescale beam
#     if calculation_scale != 1:
#         beam = reverse_resize_3d(beam[None, None, ...], transform_params)
#         beam = beam[0, 0, ...]
#         beam -= beam.min()
#         beam /= beam.max() + 1e-5

#     # Return the beam plate
#     return beam


# # -----------------------------------------------------------------------
# # ------------------------- Beam Plate Geometry -------------------------
# # -----------------------------------------------------------------------
# # All Beam Plate Geometry functions are adapted from the GDP-HMM-2025 Competition Github
# # https://github.com/RiqiangGao/GDP-HMM_AAPMChallenge/tree/main

# def interpolate_point_on_line(x1, y1, z1, x2, y2, z2, y_c):
#     """
#     Returns the coordinates of point C on the line segment from (x1, y1, z1) to (x2, y2, z2)
#     with the specified y-coordinate y_c.

#     Arguments:
#         x1, y1, z1 (int): Coordinates of point A.
#         x2, y2, z2 (int): Coordinates of point B.
#         y_c (int): The y-coordinate of point C.

#     Returns:
#         x_c, y_c, z_c (int): Coordinates of point C.
#     """

#     # Calculate the ratio of y_c relative to the total y-distance between points A and B
#     if y2 == y1:
#         y2 += 1
#     ratio = (y_c - y1) / (y2 - y1)
    
#     # Use linear interpolation to find the corresponding x and z coordinates
#     x_c = x1 + ratio * (x2 - x1)
#     z_c = z1 + ratio * (z2 - z1)
    
#     # Convert to int
#     x_c, y_c, z_c = int(x_c), int(y_c), int(z_c)

#     # Return point
#     return x_c, y_c, z_c

# def interpolate_line(x1, y1, z1, x2, y2, z2, y_c):
#     """
#     Returns all the coordinates along the line segment from (x1, y1, z1) to (x2, y2, z2).

#     Arguments:
#         x1, y1, z1 (int): Coordinates of point A.
#         x2, y2, z2 (int): Coordinates of point B.
#         y_c (int): The y-coordinate of point C.

#     Returns:
#         coordinates (list): List of tuples representing the coordinates along the line segment.
#     """

#     # Calculate the distance between the points
#     x_c, y_c, z_c = interpolate_point_on_line(x1, y1, z1, x2, y2, z2, y_c)
#     length = max(abs(x_c - x1), abs(y_c - y1), abs(z_c - z1))
    
#     # Generate linearly spaced coordinates between the points
#     x_coords = np.linspace(x1, x_c, length + 1)
#     y_coords = np.linspace(y1, y_c, length + 1)
#     z_coords = np.linspace(z1, z_c, length + 1)
    
#     # Round coordinates to integers
#     x_coords = np.round(x_coords).astype(int)
#     y_coords = np.round(y_coords).astype(int)
#     z_coords = np.round(z_coords).astype(int)
    
#     # Combine coordinates into tuples
#     coordinates = [(x, y, z) for x, y, z in zip(x_coords, y_coords, z_coords)]
    
#     # Return the coordinates
#     return coordinates

# def get_source_from_angle(isocenter, angle, spacing, r=1000):
#     """
#     Gets the source point from the isocenter and angle.

#     Arguments:
#         isocenter (tuple): Isocenter coordinates (z, y, x).
#         angle (float): Gantry angle in degrees.
#         spacing (tuple): Voxel spacing (z, y, x).
#         r (float): Distance in voxels from the isocenter to the source point.

#     Returns:
#         points (tuple): Coordinates of the source point (z, y, x).
#     """

#     # Correct the angle to match the coordinate system
#     angle = angle - 90 
    
#     # Get points
#     points = (
#         isocenter[0],
#         int(isocenter[1] + r * np.sin(np.deg2rad(angle)) / spacing[1]),
#         int(isocenter[2] + r * np.cos(np.deg2rad(angle)) / spacing[2]),

#     )

#     # Return points
#     return points

# def get_nonzero_coordinates(binary_mask):
#     """
#     Returns the coordinates of non-zero values in a binary mask.
#     """
#     return list(zip(*np.nonzero(binary_mask)))

# def get_surface_coordinates(binary_mask):
#     """
#     Get the surface coordinates of a binary mask by comparing the mask with its eroded version.
#     """

#     # Get nonzero coordinates
#     nonzero_coords = get_nonzero_coordinates(binary_mask)

#     # Initialize surface coordinates list
#     surface_coords = []

#     # Loop over nonzero coordinates
#     for x,y,z in nonzero_coords:
#         if (
#             x == 0
#             or y == 0
#             or z == 0
#             or x == binary_mask.shape[0] - 1
#             or y == binary_mask.shape[1] - 1
#             or z == binary_mask.shape[2] - 1
#         ): 
#             # Add point if it is on the edge of the mask
#             surface_coords.append((x, y, z))
#         elif (
#             binary_mask[x-1, y, z] == 0
#             or binary_mask[x+1, y, z] == 0
#             or binary_mask[x, y-1, z] == 0
#             or binary_mask[x, y+1, z] == 0
#             or binary_mask[x, y, z-1] == 0
#             or binary_mask[x, y, z+1] == 0
#         ):
#             # Add point if it is adjacent to a zero value
#             surface_coords.append((x, y, z))

#     # Return the surface coordinates
#     return surface_coords

# def get_per_beamplate(shape, surface_coords, isocenter, spacing, gantry_angle, with_distance=True):
#     """
#     Get the beam plate for a given angle.
#     """

#     # Initialize the beam plate
#     beam_plate = np.zeros(shape, dtype=bool)

#     # Get source location
#     source = get_source_from_angle(isocenter, gantry_angle, spacing)

#     # Get all line points between source and surface
#     all_points = []
#     for point in surface_coords:
#         # Correct y coordinate
#         y_c = 0 if source[1] > point[1] else shape[1] - 1
#         # Get points on line
#         line_points = interpolate_line(source[0], source[1], source[2], point[0], point[1], point[2], y_c=y_c)
#         # Filter out points
#         line_points = [p for p in line_points if (0<=p[0]<shape[0]) and (0<=p[1]<shape[1]) and (0<=p[2]<shape[2])]
#         # Add line to points
#         all_points.extend(line_points)

#     # Fill mask
#     for (x, y, z) in set(all_points):
#         beam_plate[x, y, z] = 1
#     # all_points = list(set(all_points))
#     # points = np.array(all_points, dtype=int)
#     # flat_indices = np.ravel_multi_index(points.T, beam_plate.shape, mode='clip')
#     # beam_plate.flat[flat_indices] = 1
    
#     # Fill holes in the mask TODO: Try beam_plate = ndimage.binary_closing(beam_plate, structure=np.ones((3,3,3)))
#     beam_plate = ndimage.binary_dilation(beam_plate, structure=np.ones((4,4,4)))
#     beam_plate = ndimage.binary_erosion(beam_plate, structure=np.ones((3,3,3)))

#     # Convert to float
#     beam_plate = beam_plate.astype(float)

#     # If with_distance is True, apply distance weighting
#     if with_distance:
#         # Create a meshgrid for the coordinates
#         zz, yy, xx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
#         # Get distances from source to point
#         dist = (
#             (zz - source[0])**2 
#             + (yy - source[1])**2 
#             + (xx - source[2])**2
#         )
#         # Get normalization factor, which is the distance from source to isocenter
#         norm = (
#             (isocenter[0] - source[0])**2 
#             + (isocenter[1] - source[1])**2 
#             + (isocenter[2] - source[2])**2
#         )
#         # Apply distance weighting
#         beam_plate = beam_plate * norm / (dist + 1e-5)

#     # Return the beam plate
#     return beam_plate

# def get_allbeam_plate(PTV_mask, isocenter, spacing, angles, with_distance=True):
#     """
#     Get the beam plate for all angles.

#     Arguments:
#         PTV_mask (numpy.ndarray): 3D numpy array of the PTV mask.
#         isocenter (tuple): Isocenter coordinates (z, y, x).
#         spacing (tuple): Voxel spacing (z, y, x).
#         angles (list): List of gantry angles in degrees.
#         with_distance (bool): Whether to include distance weighting.

#     Returns:
#         all_beam_plate (numpy.ndarray): 3D numpy array of the beam plate for all angles.
#     """

#     # Extract information
#     shape = PTV_mask.shape
#     surface_coords = get_surface_coordinates(PTV_mask)

#     # Initialize the beam plate
#     all_beam_plate = np.zeros(shape, dtype=float)

#     # Loop over angles
#     for angle in angles:
#         t = time.time()
#         all_beam_plate += get_per_beamplate(shape, surface_coords, isocenter, spacing, angle, with_distance=with_distance)
#         print(f'angle={angle}; time={time.time() - t}')

#     # Normalize the beam plate
#     all_beam_plate -= all_beam_plate.min()
#     all_beam_plate /= all_beam_plate.max()

#     # Return the beam plate
#     return all_beam_plate


# # ---------------------------------------------------------------------
# # ------------------------- Image Manipulation ------------------------
# # ---------------------------------------------------------------------

# # Define function to resize 3D image
# def resize_image_3d(image, target_shape, fill_value=0, return_params=False):
#     """
#     Resize a 3D image to the target shape while maintaining aspect ratio.
#     Automatically handles boolean data by using nearest-neighbor interpolation.

#     Args:
#         image (torch.Tensor): 3D image tensor of shape (B, C, D, H, W)
#         target_shape (tuple): Desired shape (D_target, H_target, W_target)
#         fill_value (int): Value to use for padding.
#         return_params (bool): If True, return parameters for inverse transformation.

#     Returns:
#         torch.Tensor: Resized image tensor of shape (B, C, D_target, H_target, W_target)
#         dict: Parameters for inverse transformation (if return_params is True).
#     """

#     # Convert to torch.Tensor if not already
#     is_numpy = isinstance(image, np.ndarray)
#     if is_numpy:
#         image = torch.from_numpy(image)

#     # Get image shape
#     _, _, D, H, W = image.shape
#     D_target, H_target, W_target = target_shape

#     # Determine interpolation mode based on dtype
#     is_boolean = image.dtype == torch.bool
#     interp_mode = 'nearest' if is_boolean else 'trilinear'
#     interp_align = False if interp_mode == 'trilinear' else None

#     # Compute uniform scale factor
#     scale = min(D_target / D, H_target / H, W_target / W)
#     new_size = (int(D * scale), int(H * scale), int(W * scale))

#     # Resize the image
#     resized_image = F.interpolate(
#         image.float(), 
#         size=new_size, 
#         mode=interp_mode, 
#         align_corners=interp_align
#     )
#     if is_boolean:
#         resized_image = resized_image.bool()

#     # Compute padding
#     pad_d = (D_target - new_size[0]) / 2
#     pad_h = (H_target - new_size[1]) / 2
#     pad_w = (W_target - new_size[2]) / 2
#     pad = (int(pad_w), int(pad_w + 0.5), int(pad_h), int(pad_h + 0.5), int(pad_d), int(pad_d + 0.5))

#     # Apply padding with the correct default value
#     image_new = F.pad(resized_image, pad, mode='constant', value=fill_value)

#     # Convert to numpy if original was numpy
#     if is_numpy:
#         image_new = image_new.numpy()

#     # Check return mode
#     if return_params:
#         # Store parameters for inverse transformation
#         transform_params = {
#             "original_shape": (D, H, W),
#             "scale": scale,
#             "pad": pad
#         }
#         # Return image and transform parameters
#         return image_new, transform_params
#     else:
#         # Return image
#         return image_new

# # Define function to reverse resize 3D image
# def reverse_resize_3d(image, transform_params):
#     """
#     Reverse the resize and padding operation.

#     Args:
#         image (torch.Tensor): Padded image tensor of shape (B, C, D_target, H_target, W_target)
#         transform_params (dict): Parameters from forward transform (original shape, scale, padding)

#     Returns:
#         torch.Tensor: Restored original-sized image.
#     """

#     # Convert to torch.Tensor if not already
#     is_numpy = isinstance(image, np.ndarray)
#     if is_numpy:
#         image = torch.from_numpy(image)

#     # Get variables
#     _, _, D_target, H_target, W_target = image.shape
#     D, H, W = transform_params["original_shape"]
#     pad = transform_params["pad"]

#     # Remove padding
#     unpadded_image = image[:, :, pad[4]:D_target-pad[5], pad[2]:H_target-pad[3], pad[0]:W_target-pad[1]]

#     # Determine interpolation mode
#     is_boolean = image.dtype == torch.bool
#     interp_mode = 'nearest' if is_boolean else 'trilinear'
#     interp_align = False if interp_mode == 'trilinear' else None

#     # Resize back to original shape
#     restored_image = F.interpolate(
#         unpadded_image.float(), 
#         size=(D, H, W), 
#         mode=interp_mode, 
#         align_corners=interp_align
#     )
#     if is_boolean:
#         restored_image = restored_image.bool()

#     # Convert to numpy if original was numpy
#     if is_numpy:
#         restored_image = restored_image.numpy()

#     # Return output
#     return restored_image












# # Test on one patient
# if __name__ == '__main__':




#     # Done
#     print('done')

