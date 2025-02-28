
# Import libraries
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset

# Get config variables
with open('config.json', 'r') as f:
    config = json.load(f)
    PATH_DATA = config['PATH_DATA']
    PATH_METADATA = config['PATH_METADATA']

# Define global variables
BAD_FILES = [
    '0522c0009+9Ag+MOS_29166.npz',
]
HaN_OAR_LIST = [
    'Cochlea_L', 
    'Cochlea_R', 
    'Eyes', 
    'Lens_L', 
    'Lens_R', 
    'OpticNerve_L', 
    'OpticNerve_R', 
    'Chiasim', 
    'LacrimalGlands', 
    'BrachialPlexus', 
    'Brain', 
    'BrainStem_03', 
    'Esophagus', 
    'Lips', 
    'Lungs', 
    'Trachea', 
    'Posterior_Neck', 
    'Shoulders', 
    'Larynx-PTV', 
    'Mandible-PTV', 
    'OCavity-PTV', 
    'ParotidCon-PTV', 
    'Parotidlps-PTV', 
    'Parotids-PTV', 
    'PharConst-PTV', 
    'Submand-PTV', 
    'SubmandL-PTV', 
    'SubmandR-PTV', 
    'Thyroid-PTV', 
    'SpinalCord_05', 
]
Lung_OAR_LIST = [
    'PTV_Ring.3-2', 
    'Total Lung-GTV', 
    'SpinalCord', 
    'Heart', 
    'LAD', 
    'Esophagus', 
    'BrachialPlexus', 
    'GreatVessels', 
    'Trachea', 
    'Body_Ring0-3', 
]
HaN_OAR_DICT = {key: i for i, key in enumerate(HaN_OAR_LIST)}
Lung_OAR_DICT = {key: i for i, key in enumerate(Lung_OAR_LIST)}


# Create dataset class
class GDPDataset(Dataset):
    def __init__(self, 
        treatment, shape=None, scale=None, return_dose=True, 
        down_HU=-1000, up_HU=1000, denom_norm_HU=500, dose_div_factor=1,
        augment=False,
    ):
        super(GDPDataset, self).__init__()

        # Get treatment specific variables
        if treatment.lower() not in ['han', 'lung']:
            raise ValueError("Treatment must be either 'HaN' or 'Lung'")
        elif treatment.lower() == 'han':
            path_data = os.path.join(PATH_DATA, 'han/train/')
            oar_list = HaN_OAR_LIST
            oar_dict = HaN_OAR_DICT
        elif treatment.lower() == 'lung':
            path_data = os.path.join(PATH_DATA, 'lung/train/')
            oar_list = Lung_OAR_LIST
            oar_dict = Lung_OAR_DICT
        
        # Load dose dictionary
        if return_dose:
            dose_dict = json.load(open(os.path.join(PATH_METADATA, 'PTV_DICT.json'), 'r'))

        # Set attributes
        self.treatment = treatment              # Treatment type (HaN or Lung)
        self.shape = shape                      # Shape of output data
        self.scale = scale                      # Scale of output data
        self.return_dose = return_dose          # Whether to return dose data
        self.down_HU = down_HU                  # Lower bound for HU values
        self.up_HU = up_HU                      # Upper bound for HU values
        self.denom_norm_HU = denom_norm_HU      # Denominator for normalizing HU values
        self.dose_div_factor = dose_div_factor  # Division factor for dose normalization
        self.augment = augment                  # Whether to augment data
        self.path_data = path_data              # Path to directory containing data files
        self.dose_dict = dose_dict              # Dictionary of dose data
        self.oar_list = oar_list                # List of organ-at-risk (OAR) names
        self.oar_dict = oar_dict                # Dictionary of OAR names and indices

        # Get list of files
        self.files = os.listdir(self.path_data)
        self.files = [f for f in self.files if f not in BAD_FILES]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        # Get file
        file = self.files[idx]

        # Get patient ID
        ID = self.files[idx].split('/')[-1].replace('.npz', '')
        PatientID = ID.split('+')[0]

        # Load data dictionary
        data_npz = np.load(os.path.join(self.path_data, file), allow_pickle=True)
        data_dict = dict(data_npz)['arr_0'].item()

        # Load CT scan
        ct = data_dict['img']
        ct = np.clip(ct, self.down_HU, self.up_HU) / self.denom_norm_HU  # Clip and normalize HU values
        ct = np.expand_dims(ct, axis=0)  # Add channel dimension

        # Load beam plate
        beam = data_dict['beam_plate']
        beam = (beam - beam.min()) / (beam.max() - beam.min())  # Normalize beam plate
        beam = np.expand_dims(beam, axis=0)  # Add channel dimension

        # Load PTVs (initialize as zeros)
        ptvs = np.zeros((3, *ct.shape[1:]), dtype=np.float32)
        for i, key in enumerate(['PTV_High', 'PTV_Mid', 'PTV_Low']):
            if key in self.dose_dict[PatientID]:  # Check if PTV exists in dose_dict
                opt_name = self.dose_dict[PatientID][key]['OPTName']
                ptv_dose = self.dose_dict[PatientID][key]['PDose']
                if opt_name in data_dict:  # Check if mask exists in data_dict
                    ptv_mask = data_dict[opt_name]
                    ptvs[i] = ptv_mask * ptv_dose / self.dose_div_factor

        # Load organ-at-risk (OAR) data
        oars = np.zeros((len(self.oar_list), *ct.shape[1:]), dtype=bool)
        for oar in self.oar_list:
            if oar in data_dict:
                oar_data = data_dict[oar]
                oars[self.oar_dict[oar]] = oar_data
        
        # Load body mask
        body = data_dict['Body']
        body = np.expand_dims(body, axis=0)  # Add channel dimension

        # Load dose
        if self.return_dose:
            # Get dose data
            dose = data_dict['dose']
            # Get dose parameters
            dose_scale = data_dict['dose_scale']
            dose_div_factor = self.dose_div_factor
            dose_ptvhigh_opt = self.dose_dict[PatientID]['PTV_High']['OPTName']
            dose_ptvhigh_dose = self.dose_dict[PatientID]['PTV_High']['PDose']
            dose_ptvhigh_mask = data_dict[dose_ptvhigh_opt].astype('bool')
            # Normalize using D97 of PTV_High
            dose = dose * dose_scale
            norm_scale = dose_ptvhigh_dose / (np.percentile(dose[dose_ptvhigh_mask], 3) + 1e-5)
            dose = dose * norm_scale / dose_div_factor
            dose = np.clip(dose, 0, dose_ptvhigh_dose * 1.2)
            # Add channel dimension
            dose = np.expand_dims(dose, axis=0)

        # Rescale data
        if self.scale is not None:
            downsample_factor = int(1/self.scale)
            ct = ct[:, ::downsample_factor, ::downsample_factor, ::downsample_factor]
            beam = beam[:, ::downsample_factor, ::downsample_factor, ::downsample_factor]
            ptvs = ptvs[:, ::downsample_factor, ::downsample_factor, ::downsample_factor]
            oars = oars[:, ::downsample_factor, ::downsample_factor, ::downsample_factor]
            body = body[:, ::downsample_factor, ::downsample_factor, ::downsample_factor]
            if self.return_dose:
                dose = dose[:, ::downsample_factor, ::downsample_factor, ::downsample_factor]

        # Reshape data
        if self.shape is not None:
            # Get shape info
            shape_old = np.array(ct.shape[1:])  # Exclude channel dimension
            shape_new = np.array(self.shape)  # Target shape
            shape_delta = shape_new - shape_old
            # Pad image to target shape
            pad_x = (max(0, shape_delta[0] // 2), max(0, shape_delta[0] - shape_delta[0] // 2))
            pad_y = (max(0, shape_delta[1] // 2), max(0, shape_delta[1] - shape_delta[1] // 2))
            pad_z = (max(0, shape_delta[2] // 2), max(0, shape_delta[2] - shape_delta[2] // 2))
            # Apply padding
            ct = np.pad(ct, ((0, 0), pad_x, pad_y, pad_z), mode='constant', constant_values=ct.min())
            beam = np.pad(beam, ((0, 0), pad_x, pad_y, pad_z), mode='constant')
            ptvs = np.pad(ptvs, ((0, 0), pad_x, pad_y, pad_z), mode='constant')
            oars = np.pad(oars, ((0, 0), pad_x, pad_y, pad_z), mode='constant')
            body = np.pad(body, ((0, 0), pad_x, pad_y, pad_z), mode='constant')
            if self.return_dose:
                dose = np.pad(dose, ((0, 0), pad_x, pad_y, pad_z), mode='constant')
            # Cropping centered at center
            center = np.array(ct.shape[1:]) // 2
            slice_x = slice(center[0] - shape_new[0] // 2, center[0] + shape_new[0] // 2)
            slice_y = slice(center[1] - shape_new[1] // 2, center[1] + shape_new[1] // 2)
            slice_z = slice(center[2] - shape_new[2] // 2, center[2] + shape_new[2] // 2)
            # Apply cropping
            ct = ct[:, slice_x, slice_y, slice_z]
            beam = beam[:, slice_x, slice_y, slice_z]
            ptvs = ptvs[:, slice_x, slice_y, slice_z]
            oars = oars[:, slice_x, slice_y, slice_z]
            body = body[:, slice_x, slice_y, slice_z]
            if self.return_dose:
                dose = dose[:, slice_x, slice_y, slice_z]

        # # Normalize data
        # ct = (ct - ct.mean()) / ct.std()
        # beam = (beam - beam.mean()) / beam.std()
        # ptvs = (ptvs - ptvs.mean()) / ptvs.std()
        # oars = (oars - oars.mean()) / oars.std()
        # body = (body - body.mean()) / body.std()
        # if self.return_dose:
        #     dose = (dose - dose.mean()) / dose.std()

        # Convert to torch tensors
        ct = torch.tensor(ct, dtype=torch.float32)
        beam = torch.tensor(beam, dtype=torch.float32)
        ptvs = torch.tensor(ptvs, dtype=torch.float32)
        oars = torch.tensor(oars, dtype=torch.float32)
        body = torch.tensor(body, dtype=torch.float32)
        if self.return_dose:
            dose = torch.tensor(dose, dtype=torch.float32)

        # Augment data
        if self.augment:
            # Apply random flip
            for dim in [1, 2, 3]:
                if np.random.rand() > 0.5:
                    ct = torch.flip(ct, (dim,))
                    beam = torch.flip(beam, (dim,))
                    ptvs = torch.flip(ptvs, (dim,))
                    oars = torch.flip(oars, (dim,))
                    body = torch.flip(body, (dim,))
                    if self.return_dose:
                        dose = torch.flip(dose, (dim,))
            # Apply random 90 degree rotation around z-axis (dim 1)
            if np.random.rand() > 0.5:
                k = np.random.randint(1, 4)
                ct = torch.rot90(ct, k, (2, 3))
                beam = torch.rot90(beam, k, (2, 3))
                ptvs = torch.rot90(ptvs, k, (2, 3))
                oars = torch.rot90(oars, k, (2, 3))
                body = torch.rot90(body, k, (2, 3))
                if self.return_dose:
                    dose = torch.rot90(dose, k, (2, 3))

        # Return data
        if self.return_dose:
            return ct, beam, ptvs, oars, body, dose
        else:
            return ct, beam, ptvs, oars, body

        

# Example usage
if __name__ == "__main__":

    # Create dataset
    dataset = GDPDataset(
        treatment='Lung', 
        # shape=(128, 128, 128),
        return_dose=True,
    )

    # Get first item
    ct, beam, ptvs, oars, body, dose = dataset[0]

    # # Plot data
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1)
    # plt.ion()
    # plt.show()
    # z_slice = ct.shape[1] // 2
    # ax.imshow(ct[0, z_slice], cmap='gray')
    # plt.tight_layout()
    # plt.pause(0.1)
    # plt.savefig('_image.png')
    # plt.close()

    # Loop over dataset
    print('Looping over dataset')
    for i in range(len(dataset)):
        # if i % 10 == 0:
        #     print(f'-- {i}/{len(dataset)} --')
        ct, beam, ptvs, oars, body, dose = dataset[i]
        print(i, ct.shape)

    # Done
    print("Done")


