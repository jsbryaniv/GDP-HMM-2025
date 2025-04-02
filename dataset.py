
# Import libraries
import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

# Import local
from config import *
from utils import augment_data_3d

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
All_OAR_LIST = sorted(list(set(HaN_OAR_LIST + Lung_OAR_LIST)))
HaN_OAR_DICT = {key: i for i, key in enumerate(HaN_OAR_LIST)}
Lung_OAR_DICT = {key: i for i, key in enumerate(Lung_OAR_LIST)}
All_OAR_DICT = {key: i for i, key in enumerate(All_OAR_LIST)}


# Create dataset class
class GDPDataset(Dataset):
    def __init__(self, treatment, validation_set=False, augment=True):
        super(GDPDataset, self).__init__()

        # Check inputs
        if validation_set:
            augment = False  # Turn off augmentation for validation set
        if treatment is None:
            treatment = 'All'  # Get treatment
        
        # Load dose dictionary
        dose_dict = json.load(open(os.path.join(PATH_METADATA, 'PTV_DICT.json'), 'r'))

        # Get OAR info
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

        # Get constants
        n_channels = 6 + len(oar_list)  # Number of input channels
        scan_min=-1000
        scan_max=1000 
        scan_norm=500

        # Set attributes
        self.treatment = treatment              # Treatment type (HaN, Lung, or All)
        self.validation_set = validation_set    # Validation set flag
        self.dose_dict = dose_dict              # Dictionary of dose data
        self.oar_list = oar_list                # List of organ-at-risk (OAR) names
        self.oar_dict = oar_dict                # Dictionary of OAR names and indices
        self.n_channels = n_channels            # Number of channels
        self.scan_min = scan_min                # Lower bound for HU values
        self.scan_max = scan_max                # Upper bound for HU values
        self.scan_norm = scan_norm              # Denominator for normalizing HU values
        self.augment = augment                  # Augmentation flag

        # Set up augmentor
        self.augmentor = augment_data_3d if augment else None

        # Get list of files
        path_train_or_val = 'valid_nodose' if validation_set else 'train'
        if (treatment.lower() == 'han') or (treatment.lower() == 'lung'):
            # Get either HaN or Lung files
            path_data = os.path.join(PATH_DATA, treatment.lower(), path_train_or_val)
            files = [os.path.join(path_data, f) for f in os.listdir(path_data) if f not in BAD_FILES]
        elif self.treatment.lower() == 'all':
            # Get all files
            path_han = os.path.join(PATH_DATA, 'han', path_train_or_val)
            path_lung = os.path.join(PATH_DATA, 'lung', path_train_or_val)
            files_han = [os.path.join(path_han, f) for f in os.listdir(path_han) if f not in BAD_FILES]
            files_lung = [os.path.join(path_lung, f) for f in os.listdir(path_lung) if f not in BAD_FILES]
            files = files_han + files_lung
        files = sorted(files)
        self.files = files

    def get_config(self):
        return {
            'treatment': self.treatment,
            'validation_set': self.validation_set,
            'augment': self.augment,
        }

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
        scan = data_dict['img']
        scan = np.clip(scan, self.scan_min, self.scan_max) / self.scan_norm  # Clip and normalize HU values
        scan = np.expand_dims(scan, axis=0)  # Add channel dimension

        # Load beam plate
        beam = data_dict['beam_plate']
        beam = (beam - beam.min()) / (beam.max() - beam.min())  # Normalize beam plate
        beam = np.expand_dims(beam, axis=0)  # Add channel dimension

        # Load PTVs (initialize as zeros)
        ptvs = np.zeros((3, *scan.shape[1:]), dtype=np.float32)
        for i, key in enumerate(['PTV_High', 'PTV_Mid', 'PTV_Low']):
            if key in self.dose_dict[PatientID]:  # Check if PTV exists in dose_dict
                opt_name = self.dose_dict[PatientID][key]['OPTName']
                ptv_dose = self.dose_dict[PatientID][key]['PDose']
                if opt_name in data_dict:  # Check if mask exists in data_dict
                    ptv_mask = data_dict[opt_name]
                    ptvs[i] = ptv_mask * ptv_dose

        # Load organ-at-risk (OAR) data
        oars = np.zeros((len(self.oar_list), *scan.shape[1:]), dtype=bool)
        for oar in self.oar_list:
            if oar in data_dict:
                oar_data = data_dict[oar]
                oars[self.oar_dict[oar]] = oar_data
        
        # Load body mask
        body = data_dict['Body']
        body = np.expand_dims(body, axis=0)  # Add channel dimension

        # Load dose
        if self.validation_set:
            dose = None
        else:
            # Get dose data
            dose = data_dict['dose']
            # Get dose parameters
            dose_scale = data_dict['dose_scale']
            dose_ptvhigh = self.dose_dict[PatientID]['PTV_High']['PDose']
            dose_ptvhigh_name = self.dose_dict[PatientID]['PTV_High']['OPTName']
            dose_ptvhigh_mask = data_dict[dose_ptvhigh_name].astype('bool')
            # Normalize using D97 of PTV_High
            dose = dose * dose_scale
            norm_scale = dose_ptvhigh / (np.percentile(dose[dose_ptvhigh_mask], 3) + 1e-5)
            dose = dose * norm_scale
            dose = np.clip(dose, 0, dose_ptvhigh * 1.2)
            # Add channel dimension
            dose = np.expand_dims(dose, axis=0)

        # Convert to torch tensors
        scan = torch.tensor(scan, dtype=torch.float32)
        beam = torch.tensor(beam, dtype=torch.float32)
        ptvs = torch.tensor(ptvs, dtype=torch.float32)
        oars = torch.tensor(oars, dtype=torch.bool)
        body = torch.tensor(body, dtype=torch.bool)
        if self.validation_set:
            dose = None
        else:
            dose = torch.tensor(dose, dtype=torch.float32)

        # Augment data
        if self.augment:
            scan, beam, ptvs, oars, body, dose = self.augmentor(
                scan, beam, ptvs, oars, body, 
                targets=dose,
            )

        # Return data
        return scan, beam, ptvs, oars, body, dose


# Define collate function
def collate_gdp(batch):
    """
    Collates a batch of tensors by padding them to the max shape in the batch.
    Works for 3D tensors of arbitrary size.
    """

    # Get shape from first element
    batch_size = len(batch)
    scan0, beam0, ptvs0, oars0, body0, dose0 = batch[0]
    has_dose = dose0 is not None
    n_channels_scan = scan0.shape[0]
    n_channels_beam = beam0.shape[0]
    n_channels_ptvs = ptvs0.shape[0]
    n_channels_oars = oars0.shape[0]
    n_channels_body = body0.shape[0]
    n_channels_dose = dose0.shape[0] if has_dose else 0

    # Determine max spatial size
    D, H, W = zip(*[x[0].shape[-3:] for x in batch])
    max_D, max_H, max_W = max(D), max(H), max(W)

    # Preallocate tensors
    scan_batch = torch.ones((batch_size, n_channels_scan, max_D, max_H, max_W), dtype=torch.float32) * scan0.min().item()
    beam_batch = torch.zeros((batch_size, n_channels_beam, max_D, max_H, max_W), dtype=torch.float32)
    ptvs_batch = torch.zeros((batch_size, n_channels_ptvs, max_D, max_H, max_W), dtype=torch.float32)
    oars_batch = torch.zeros((batch_size, n_channels_oars, max_D, max_H, max_W), dtype=torch.bool)
    body_batch = torch.zeros((batch_size, n_channels_body, max_D, max_H, max_W), dtype=torch.bool)
    if has_dose:
        dose_batch = torch.zeros((batch_size, n_channels_dose, max_D, max_H, max_W), dtype=torch.float32)
    else:
        dose_batch = None

    # Fill batches
    for i, (scan, beam, ptvs, oars, body, dose) in enumerate(batch):
        d, h, w = scan.shape[-3:]
        d_pad = (max_D - d) // 2
        h_pad = (max_H - h) // 2
        w_pad = (max_W - w) // 2

        scan_batch[i, :, d_pad:d_pad+d, h_pad:h_pad+h, w_pad:w_pad+w] = scan
        beam_batch[i, :, d_pad:d_pad+d, h_pad:h_pad+h, w_pad:w_pad+w] = beam
        ptvs_batch[i, :, d_pad:d_pad+d, h_pad:h_pad+h, w_pad:w_pad+w] = ptvs
        oars_batch[i, :, d_pad:d_pad+d, h_pad:h_pad+h, w_pad:w_pad+w] = oars
        body_batch[i, :, d_pad:d_pad+d, h_pad:h_pad+h, w_pad:w_pad+w] = body
        if has_dose:
            dose_batch[i, :, d_pad:d_pad+d, h_pad:h_pad+h, w_pad:w_pad+w] = dose
    
    # Return output
    return scan_batch, beam_batch, ptvs_batch, oars_batch, body_batch, dose_batch



# Example usage
if __name__ == "__main__":

    # Index 733 shape = (141, 156, 259)
    # Index 734 shape = (141, 156, 259)

    # Import libraries
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    # Create dataset
    dataset = GDPDataset(
        treatment='All', 
        validation_set=False,
    )

    # Create dataloader
    batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_gdp)

    # Loop over dataloader
    for i, (scan, beam, ptvs, oars, body, dose) in enumerate(loader):
        if i % 100 == 0:
            print(f'--{i}/{len(loader)}')

        # Plot
        fig, ax = plt.subplots(1, 6)
        plt.ion()
        plt.show()
        d_slice = scan.shape[2] // 2
        for i, x in enumerate([scan, beam, ptvs, oars, body, dose]):
            ax[i].imshow(x[0, 0, d_slice], cmap='gray')
            ax[i].axis('off')
        plt.tight_layout()
        plt.pause(0.1)
        plt.savefig('_image.png', dpi=900)
        plt.close()

    # Done
    print("Done")

    # # Initialize lists
    # shapes_val = []
    # dose_val = []
    # ptvs_val = []
    # isVMAT_val = []

    # # Loop over dataset
    # print('Looping over dataset')
    # for i in range(len(dataset)):
    #     if i % 100 == 0:
    #         print(f'--{i}/{len(dataset)}')

    #     # Get data
    #     # scan, beam, ptvs, oars, body, dose = dataset[i]
    #     scan, beam, ptvs, oars, body, dose, isVMAT = dataset[i]

    #     # Append data
    #     shapes_val.append([i for i in scan.shape[1:]])
    #     dose_val.append(dose.max() if dose is not None else None)
    #     ptvs_val.append(np.unique(ptvs.detach().cpu().numpy()))
    #     isVMAT_val.append(isVMAT)

    # # Create dataset
    # dataset = GDPDataset(
    #     treatment='All', 
    #     validation_set=False,
    # )

    # # Initialize lists
    # shapes_train = []
    # dose_train = []
    # ptvs_train = []
    # isVMAT_train = []

    # # Loop over dataset
    # print('Looping over dataset')
    # for i in range(len(dataset)):
    #     if i % 100 == 0:
    #         print(f'--{i}/{len(dataset)}')

    #     # Get data
    #     # scan, beam, ptvs, oars, body, dose = dataset[i]
    #     scan, beam, ptvs, oars, body, dose, isVMAT = dataset[i]

    #     # Append data
    #     shapes_train.append([i for i in scan.shape[1:]])
    #     dose_train.append(dose.max().item() if dose is not None else None)
    #     ptvs_train.append(np.unique(ptvs.detach().cpu().numpy()))
    #     isVMAT_train.append(isVMAT)


    # # Get max shape of validation set
    # max_shape_val = (0, 0, 0)
    # for shape in shapes_val:
    #     max_shape_val = tuple([max(s, m) for s, m in zip(shape, max_shape_val)])
    # print('Max shape validation set:', max_shape_val)  # (138, 148, 229)

    # # Get max shape of training set
    # max_shape_train = (0, 0, 0)
    # for shape in shapes_train:
    #     max_shape_train = tuple([max(s, m) for s, m in zip(shape, max_shape_train)])
    # print('Max shape training set:', max_shape_train)  # (165, 156, 259)

    # # Get unique PTV values in validation set
    # ptvs_val_unique = set()
    # for ptvs in ptvs_val:
    #     ptvs_val_unique.update(ptvs)
    # ptvs_val_unique = sorted(list(ptvs_val_unique))
    # print('Unique PTV values in validation set:', ptvs_val_unique)
    # # [0.0, 50.0, 50.4, 52.1, 52.5, 53.4, 53.6, 53.9, 54.0, 54.5, 55.6, 56.0, 56.1, 57.0, 58.0, 59.0, 59.4, 59.5, 59.8, 60.0, 62.7, 63.0, 65.0, 66.0, 68.0, 70.0]

    # # Get unique PTV values in training set
    # ptvs_train_unique = set()
    # for ptvs in ptvs_train:
    #     ptvs_train_unique.update(ptvs)
    # ptvs_train_unique = sorted(list(ptvs_train_unique))
    # print('Unique PTV values in training set:', ptvs_train_unique)  
    # # [0.0, 10.0, 28.0, 40.2, 42.8, 43.2, 44.8, 45.0, 46.9, 47.9, 49.3, 49.4, 50.0, 50.4, 50.5, 50.6, 50.9, 51.4, 51.6, 51.7, 51.9, 52.0, 52.3, 52.4, 52.5, 52.6, 52.8, 52.9, 53.0, 53.3, 54.0, 54.1, 54.5, 54.8, 55.0, 55.1, 55.8, 56.0, 56.1, 56.7, 57.0, 57.1, 57.6, 58.0, 58.1, 59.0, 59.2, 59.4, 59.5, 59.6, 60.0, 60.1, 61.8, 62.0, 62.7, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 72.0]

    # # Get fraction of VMAT in validation set
    # isVMAT_val_frac = sum(isVMAT_val) / len(isVMAT_val)
    # print('Fraction of VMAT in validation set:', isVMAT_val_frac)  # 0.4044943820224719

    # # Get fraction of VMAT in training set
    # isVMAT_train_frac = sum(isVMAT_train) / len(isVMAT_train)
    # print('Fraction of VMAT in training set:', isVMAT_train_frac)  # 0.4278762599930483




