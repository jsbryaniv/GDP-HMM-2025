
# Import libraries
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset


# Define organ-at-risk (OAR) dictionaries
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
        treatment, path_data, path_dose_dict=None, return_dose=True, 
        down_HU=-1000, up_HU=1000, denom_norm_HU=500, dose_div_factor=10,
    ):
        super(GDPDataset, self).__init__()

        # Check inputs
        if treatment.lower() not in ['han', 'lung']:
            raise ValueError("Treatment must be either 'HaN' or 'Lung'")
        if return_dose and path_dose_dict is None:
            raise ValueError("Path to dose dictionary must be provided if 'return_dose' is True")
        
        # Load dose dictionary
        if return_dose:
            dose_dict = json.load(open(path_dose_dict, 'r'))

        # Set attributes
        self.path = path_data                  # Path to directory containing data files
        self.treatment = treatment             # Treatment type (HaN or Lung)
        self.dose_dict = dose_dict             # Dictionary of dose data
        self.return_dose = return_dose         # Whether to return dose data
        self.down_HU = down_HU                 # Lower bound for HU values
        self.up_HU = up_HU                     # Upper bound for HU values
        self.denom_norm_HU = denom_norm_HU     # Denominator for normalizing HU values
        self.dose_div_factor = dose_div_factor # Division factor for dose normalization

        # Get list of files
        self.files = os.listdir(self.path)

        # Set organ-at-risk (OAR) information
        if self.treatment.lower() == 'han':
            self.OAR_LIST = HaN_OAR_LIST
            self.OAR_DICT = HaN_OAR_DICT
        elif self.treatment.lower() == 'lung':
            self.OAR_LIST = Lung_OAR_LIST
            self.OAR_DICT = Lung_OAR_DICT

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        # Get file
        file = self.files[idx]

        # Get patient ID
        ID = self.files[idx].split('/')[-1].replace('.npz', '')
        PatientID = ID.split('+')[0]

        # Load data dictionary
        data_npz = np.load(os.path.join(self.path, file), allow_pickle=True)
        data_dict = dict(data_npz)['arr_0'].item()

        # Load CT scan
        ct = data_dict['img']
        ct = np.clip(ct, self.down_HU, self.up_HU) / self.denom_norm_HU  # Clip and normalize HU values
        ct = np.expand_dims(ct, axis=0)  # Add channel dimension

        # Load beam plate
        beam = data_dict['beam_plate']
        beam = np.expand_dims(beam, axis=0)  # Add channel dimension

        # Load organ-at-risk (OAR) data
        oars = np.zeros((len(self.OAR_LIST), *ct.shape), dtype=bool)
        for oar in self.OAR_LIST:
            if oar in data_dict:
                oar_data = data_dict[oar]
                oars[self.OAR_DICT[oar]] = oar_data

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
        oars = torch.tensor(oars, dtype=torch.float32)
        if self.return_dose:
            dose = torch.tensor(dose, dtype=torch.float32)

        # Return data
        if self.return_dose:
            return ct, beam, oars, dose
        else:
            return ct, beam, oars

        

# Example usage
if __name__ == "__main__":
    
    # Set paths
    path_data = "data/han/train"
    path_dose_dict = "data/PTV_DICT.json"

    # Create dataset
    dataset = GDPDataset(
        treatment='HaN', 
        path_data=path_data, 
        path_dose_dict=path_dose_dict, 
        return_dose=True,
    )

    # Get first item
    ct, beam, oars, dose = dataset[0]

    # Loop over dataset
    print('Looping over dataset')
    for i in range(len(dataset)):
        if i % 10 == 0:
            print(f'-- {i}/{len(dataset)} --')
        ct, beam, oars, dose = dataset[i]

    # Done
    print("Done")


