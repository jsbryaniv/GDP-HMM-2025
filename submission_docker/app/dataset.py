
# Import libraries
import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

# Define global variables
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
All_OAR_DICT = {key: i for i, key in enumerate(All_OAR_LIST)}


# Create dataset class
class GDPTestDataset(Dataset):
    def __init__(self, path_data):
        super(GDPTestDataset, self).__init__()
        
        # Load dose dictionary
        dose_dict = json.load(open(os.path.join(path_data, 'PTV_DICT.json'), 'r'))

        # Get OAR info
        oar_list = All_OAR_LIST
        oar_dict = All_OAR_DICT

        # Get constants
        n_channels = 6 + len(oar_list)  # Number of input channels
        scan_min=-1000
        scan_max=1000 
        scan_norm=500

        # Set attributes
        self.path_data = path_data              # Path to data
        self.dose_dict = dose_dict              # Dictionary of dose data
        self.oar_list = oar_list                # List of organ-at-risk (OAR) names
        self.oar_dict = oar_dict                # Dictionary of OAR names and indices
        self.n_channels = n_channels            # Number of channels
        self.scan_min = scan_min                # Lower bound for HU values
        self.scan_max = scan_max                # Upper bound for HU values
        self.scan_norm = scan_norm              # Denominator for normalizing HU values

        # Get list of files
        files = [os.path.join(path_data, f) for f in os.listdir(path_data) if f.endswith('.npz')]
        files = sorted(files)
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
        body = np.expand_dims(body, axis=0)

        # Convert to torch tensors
        scan = torch.tensor(scan, dtype=torch.float32)
        beam = torch.tensor(beam, dtype=torch.float32)
        ptvs = torch.tensor(ptvs, dtype=torch.float32)
        oars = torch.tensor(oars, dtype=torch.bool)
        body = torch.tensor(body, dtype=torch.bool)

        # Return data
        return scan, beam, ptvs, oars, body


# Define collate function
def collate_gdp(batch):
    """
    Collates a batch of tensors by padding them to the max shape in the batch.
    Works for 3D tensors of arbitrary size.
    """

    # Get shape from first element
    batch_size = len(batch)
    scan0, beam0, ptvs0, oars0, body0 = batch[0]
    n_channels_scan = scan0.shape[0]
    n_channels_beam = beam0.shape[0]
    n_channels_ptvs = ptvs0.shape[0]
    n_channels_oars = oars0.shape[0]
    n_channels_body = body0.shape[0]

    # Determine max spatial size
    D, H, W = zip(*[x[0].shape[-3:] for x in batch])
    max_D, max_H, max_W = max(D), max(H), max(W)

    # Preallocate tensors
    scan_batch = torch.ones((batch_size, n_channels_scan, max_D, max_H, max_W), dtype=torch.float32)
    beam_batch = torch.zeros((batch_size, n_channels_beam, max_D, max_H, max_W), dtype=torch.float32)
    ptvs_batch = torch.zeros((batch_size, n_channels_ptvs, max_D, max_H, max_W), dtype=torch.float32)
    oars_batch = torch.zeros((batch_size, n_channels_oars, max_D, max_H, max_W), dtype=torch.bool)
    body_batch = torch.zeros((batch_size, n_channels_body, max_D, max_H, max_W), dtype=torch.bool)

    # Fill batches
    for i, (scan, beam, ptvs, oars, body) in enumerate(batch):
        d, h, w = scan.shape[-3:]
        d_pad = (max_D - d) // 2
        h_pad = (max_H - h) // 2
        w_pad = (max_W - w) // 2

        scan_batch[i] *= scan.min()  # Set outside values to min
        scan_batch[i, :, d_pad:d_pad+d, h_pad:h_pad+h, w_pad:w_pad+w] = scan
        beam_batch[i, :, d_pad:d_pad+d, h_pad:h_pad+h, w_pad:w_pad+w] = beam
        ptvs_batch[i, :, d_pad:d_pad+d, h_pad:h_pad+h, w_pad:w_pad+w] = ptvs
        oars_batch[i, :, d_pad:d_pad+d, h_pad:h_pad+h, w_pad:w_pad+w] = oars
        body_batch[i, :, d_pad:d_pad+d, h_pad:h_pad+h, w_pad:w_pad+w] = body
    
    # Return output
    return scan_batch, beam_batch, ptvs_batch, oars_batch, body_batch



# Example usage
if __name__ == "__main__":

    # Import libraries
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    # Create dataset
    path_data = '../data/'
    dataset = GDPTestDataset(path_data)

    # Create dataloader
    batch_size = 1
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_gdp)

    # Loop over dataloader
    for i, (scan, beam, ptvs, oars, body) in enumerate(loader):
        print(f'--{i}/{len(loader)}')

        # Plot
        fig, ax = plt.subplots(1, 6)
        plt.ion()
        plt.show()
        d_slice = scan.shape[2] // 2
        for i, x in enumerate([scan, beam, ptvs, oars, body]):
            ax[i].imshow(x[0, 0, d_slice], cmap='gray')
            ax[i].axis('off')
        plt.tight_layout()
        plt.pause(0.1)
        plt.close()

    # Done
    print("Done")



