
# Import libraries
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset

# Import custom libraries
from utils import resize_image_3d

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
All_OAR_LIST = list(set(HaN_OAR_LIST + Lung_OAR_LIST))
HaN_OAR_DICT = {key: i for i, key in enumerate(HaN_OAR_LIST)}
Lung_OAR_DICT = {key: i for i, key in enumerate(Lung_OAR_LIST)}
All_OAR_DICT = {key: i for i, key in enumerate(All_OAR_LIST)}


# Create dataset class
class GDPDataset(Dataset):
    def __init__(self, 
        treatment, shape=None, return_dose=True, validation_set=False,
        down_HU=-1000, up_HU=1000, denom_norm_HU=500, dose_div_factor=1,
        augment=False,
    ):
        super(GDPDataset, self).__init__()

        # Check inputs
        if validation_set:
            return_dose = False
        if isinstance(shape, int):
            shape = (shape, shape, shape)

        # Get treatment specific variables
        if treatment.lower() not in ['han', 'lung', 'all']:
            raise ValueError("Treatment must be either 'HaN', 'Lung' or 'All'.")
        if treatment.lower() == 'han':
            oar_list = HaN_OAR_LIST
            oar_dict = HaN_OAR_DICT
        elif treatment.lower() == 'lung':
            oar_list = Lung_OAR_LIST
            oar_dict = Lung_OAR_DICT
        elif treatment.lower() == 'all':
            oar_list = All_OAR_LIST
            oar_dict = All_OAR_DICT
        
        # Load dose dictionary
        dose_dict = json.load(open(os.path.join(PATH_METADATA, 'PTV_DICT.json'), 'r'))

        # Set attributes
        self.treatment = treatment              # Treatment type (HaN or Lung)
        self.shape = shape                      # Shape of output data
        self.return_dose = return_dose          # Whether to return dose data
        self.down_HU = down_HU                  # Lower bound for HU values
        self.up_HU = up_HU                      # Upper bound for HU values
        self.denom_norm_HU = denom_norm_HU      # Denominator for normalizing HU values
        self.dose_div_factor = dose_div_factor  # Division factor for dose normalization
        self.augment = augment                  # Whether to augment data
        self.dose_dict = dose_dict              # Dictionary of dose data
        self.oar_list = oar_list                # List of organ-at-risk (OAR) names
        self.oar_dict = oar_dict                # Dictionary of OAR names and indices

        # Get list of files
        # self.files = os.listdir(self.path_data)
        # self.files = [f for f in self.files if f not in BAD_FILES]
        path_train_or_val = 'valid_nodose' if validation_set else 'train'
        if self.treatment.lower() == 'han':
            path_data = os.path.join(PATH_DATA, 'han', path_train_or_val)
            files = [os.path.join(path_data, f) for f in os.listdir(path_data) if f not in BAD_FILES]
        elif self.treatment.lower() == 'lung':
            path_data = os.path.join(PATH_DATA, 'lung', path_train_or_val)
            files = [os.path.join(path_data, f) for f in os.listdir(path_data) if f not in BAD_FILES]
        elif self.treatment.lower() == 'all':
            path_han = os.path.join(PATH_DATA, 'han', path_train_or_val)
            path_lung = os.path.join(PATH_DATA, 'lung', path_train_or_val)
            files_han = [os.path.join(path_han, f) for f in os.listdir(path_han) if f not in BAD_FILES]
            files_lung = [os.path.join(path_lung, f) for f in os.listdir(path_lung) if f not in BAD_FILES]
            files = files_han + files_lung
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        # Get file
        file = self.files[idx]

        # Get patient ID
        ID = self.files[idx].split('/')[-1].replace('.npz', '')
        PatientID = ID.split('+')[0]

        # Load data dictionary
        data_npz = np.load(file, allow_pickle=True)
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

        # Convert to torch tensors
        ct = torch.tensor(ct, dtype=torch.float32)
        beam = torch.tensor(beam, dtype=torch.float32)
        ptvs = torch.tensor(ptvs, dtype=torch.float32)
        oars = torch.tensor(oars, dtype=torch.bool)
        body = torch.tensor(body, dtype=torch.bool)
        if self.return_dose:
            dose = torch.tensor(dose, dtype=torch.float32)

        # Resize data
        if self.shape is not None:
            ct, _ = resize_image_3d(ct, self.shape)
            beam, _ = resize_image_3d(beam, self.shape)
            ptvs, _ = resize_image_3d(ptvs, self.shape)
            oars, _ = resize_image_3d(oars, self.shape)
            body, _ = resize_image_3d(body, self.shape)
            if self.return_dose:
                dose, _ = resize_image_3d(dose, self.shape)
        
        # # Normalize data
        # ct = (ct - ct.mean()) / ct.std()
        # beam = (beam - beam.mean()) / beam.std()
        # ptvs = (ptvs - ptvs.mean()) / ptvs.std()
        # oars = (oars - oars.mean()) / oars.std()
        # body = (body - body.mean()) / body.std()
        # if self.return_dose:
        #     dose = (dose - dose.mean()) / dose.std()

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

    # Import libraries
    import matplotlib.pyplot as plt

    # Create dataset
    dataset = GDPDataset(
        treatment='Lung', 
        # shape=(128, 128, 128),
        validation_set=True,
    )

    # Loop over dataset
    print('Looping over dataset')
    for i in range(len(dataset)):
        # Get data
        ct, beam, ptvs, oars, body = dataset[i][:5]
        if len(dataset[i]) > 5:
            dose = dataset[i][-1]
        print(i, ct.shape)
        # Plot data
        fig, ax = plt.subplots(1, 1)
        plt.ion()
        plt.show()
        z_slize = ct.shape[1] // 2
        ax.imshow(ct[0, z_slize].detach().cpu().numpy(), cmap='gray')
        plt.tight_layout()
        plt.pause(0.1)
        plt.savefig('_image.png')
        plt.close()

    # Done
    print("Done")




