
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import custom libraries
from losses import competition_loss, dvh_loss
from utils import resize_image_3d, reverse_resize_3d, block_mask_3d


# Create dose prediction model
class DosePredictionModel(nn.Module):
    """Dose prediction model."""
    def __init__(self, architecture, n_channels, shape=None, **kwargs):
        super(DosePredictionModel, self).__init__()

        # Check inputs
        if isinstance(shape, int):
            # Convert shape to tuple
            shape = (shape, shape, shape)  
        
        # Set attributes
        self.architecture = architecture
        self.n_channels = n_channels
        self.shape = shape
        self.kwargs = kwargs

        # Initialize model
        if architecture.lower() == "test":
            # Test model (Really lightweight Unet)
            from architectures.unet import Unet3d
            kwargs = {
                'in_channels': n_channels,
                'out_channels': 1,
                'n_blocks': 0,             # Dummy model for testing (0 blocks)
                'n_layers_per_block': 0,   # Dummy model for testing (0 layers)
                'n_features': 1,           # Dummy model for testing (1 feature)
                **kwargs,
            }
            self.model = Unet3d(**kwargs)
        elif architecture.lower() == "unet":
            # Unet3D
            from architectures.unet import Unet3d
            kwargs = {
                'in_channels': n_channels,
                'out_channels': 1,
                **kwargs,
            }
            self.model = Unet3d(**kwargs)
        elif architecture.lower() == "vit":
            # ViT3D
            from architectures.vit import ViT3d
            kwargs = {
                'in_channels': n_channels,
                'out_channels': 1,
                'shape': shape,
                **kwargs,
            }
            self.model = ViT3d(**kwargs)
        elif architecture.lower() == "moeunet":
            # MOEUnet3D
            from architectures.unet import Unet3d
            from architectures.moe import MOEWrapper3d
            kwargs = {
                'in_channels': n_channels,
                'out_channels': 1,
                **kwargs,
            }
            self.model = MOEWrapper3d(Unet3d, **kwargs)
        elif architecture.lower() == "moevit":
            # MOEViT3D
            from architectures.vit import ViT3d
            from architectures.moe import MOEWrapper3d
            kwargs = {
                'in_channels': n_channels,
                'out_channels': 1,
                'shape': shape,
                **kwargs,
            }
            self.model = MOEWrapper3d(ViT3d, **kwargs)
        elif architecture.lower() == "crossunet":
            # CrossUnetModel
            from architectures.crossunet import CrossUnetModel
            kwargs = {
                'in_channels': n_channels,
                'out_channels': 1,
                'n_cross_channels_list': [1, 4, n_channels-5],  # scan, (beam, ptvs), (oars, body)
                **kwargs,
            }
            self.model = CrossUnetModel(**kwargs)        
        elif architecture.lower() == "crossunetlight":
            # CrossUnetModel
            from architectures.crossunet import CrossUnetModel
            kwargs = {
                'in_channels': n_channels,
                'out_channels': 1,
                'n_cross_channels_list': [n_channels],
                **kwargs,
            }
            self.model = CrossUnetModel(**kwargs)
        elif architecture.lower() == "crossvit":
            # CrossViT3d
            from architectures.crossvit import CrossViT3d
            kwargs = {
                'in_channels': n_channels,
                'out_channels': 1,
                'n_cross_channels_list': [1, 4, n_channels-5],  # scan, (beam, ptvs), (oars, body)
                'shape': shape,
                **kwargs,
            }
            self.model = CrossViT3d(**kwargs)
        elif architecture.lower() == "diffunet":
            # DiffUnet3d
            from architectures.diffunet import DiffUnet3d
            kwargs = {
                'in_channels': 1,
                'n_cross_channels_list': [1, 4, n_channels-5],  # scan, (beam, ptvs), (oars, body)
                **kwargs,
            }
            self.model = DiffUnet3d(**kwargs)
        elif architecture.lower() == "diffvit":
            # DiffViT3d
            from architectures.diffvit import DiffViT3d
            kwargs = {
                'in_channels': 1,
                'n_cross_channels_list': [1, 4, n_channels-5],  # scan, (beam, ptvs), (oars, body)
                'shape': shape,
                **kwargs,
            }
            self.model = DiffViT3d(**kwargs)
        elif architecture.lower() == "diffunetlight":
            # DiffUnet3d
            from architectures.diffunet import DiffUnet3d
            kwargs = {
                'in_channels': 1,
                'n_cross_channels_list': [n_channels],
                **kwargs,
            }
            self.model = DiffUnet3d(**kwargs)
        elif architecture.lower() == "diffvitlight":
            # DiffViT3d
            from architectures.diffvit import DiffViT3d
            kwargs = {
                'in_channels': 1,
                'n_cross_channels_list': [n_channels],
                'shape': shape,
                **kwargs,
            }
            self.model = DiffViT3d(**kwargs)
        else:
            raise ValueError(f"Architecture '{architecture}' not recognized.")
        
    def get_config(self):
        return {
            'architecture': self.architecture,
            'n_channels': self.n_channels,
            'shape': self.shape,
            **self.model.get_config(),
        }
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, model_config=None, model_state_dict=None):
        """
        Load model from checkpoint.
        """

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        # Initialize model with config
        if model_config is None:
            model_config = checkpoint['model_config']
        model = cls(**model_config)
        
        # Load model state
        if model_state_dict is None:
            model_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_state_dict)
        
        # Return model
        return model
        
    def format_inputs(self, scan, beam, ptvs, oars, body):

        # Initialize transform params
        transform_params = {
            'original_shape': scan.shape[2:],
        }

        # Rescale doses
        dose_scale = ptvs.max().item() if ptvs.max() > 0 else 1
        ptvs = ptvs / dose_scale
        transform_params['dose_scale'] = dose_scale

        # Reshape inputs
        if self.shape is not None:
            ### Resize to shape ###
            scan, resize_params = resize_image_3d(scan, self.shape, fill_value=scan.min())
            beam, _ = resize_image_3d(beam, self.shape)
            ptvs, _ = resize_image_3d(ptvs, self.shape)
            oars, _ = resize_image_3d(oars, self.shape)
            body, _ = resize_image_3d(body, self.shape)
            ### Update transform params ###
            transform_params['type'] = 'resize'
            transform_params['resize'] = resize_params

        # Check architecture
        if self.architecture.lower() in ["test", "unet", "vit", "moeunet", "moevit"]:
            # Concatenate inputs
            inputs = (
                torch.cat([scan, beam, ptvs, oars, body], dim=1),
            )
        elif self.architecture.lower() in ["crossunet", "crossvit", "moecrossunet", "moecrossvit"]:
            # Separate contexts
            inputs = (
                torch.cat([scan, beam, ptvs, oars, body], dim=1).clone(),  # Main input
                scan,                                                      # Context 1  
                torch.cat([beam, ptvs], dim=1),                            # Context 2
                torch.cat([oars, body], dim=1).float(),                    # Context 3
            )
        elif self.architecture.lower() in ["crossunetlight", "crossvitlight", "moecrossunetlight", "moecrossvitlight"]:
            # Separate contexts
            inputs = (
                torch.cat([scan, beam, ptvs, oars, body], dim=1).clone(),  # Main input
                torch.cat([scan, beam, ptvs, oars, body], dim=1).clone(),  # Context
            )
        elif self.architecture.lower() in ["diffunet", "diffvit"]:
            # Separate contexts
            inputs = (
                scan,                                    # Context 1  
                torch.cat([beam, ptvs], dim=1),          # Context 2
                torch.cat([oars, body], dim=1).float(),  # Context 3
            )
        elif self.architecture.lower() in ["diffunetlight", "diffvitlight"]:
            # Separate contexts
            inputs = (
                torch.cat([scan, beam, ptvs, oars, body], dim=1),  # All context grouped together
            )

        # Return inputs
        return inputs, transform_params
    
    def format_outputs(self, x, transform_params):
        
        # Reshape prediction
        if self.shape is not None:
            resize_params = transform_params['resize']
            x = reverse_resize_3d(x, resize_params)

        # Rescale prediction
        dose_scale = transform_params['dose_scale']
        x = x * dose_scale

        # Return prediction
        return x
        
    def forward(self, scan, beam, ptvs, oars, body):
        
        # Format inputs
        inputs, transform_params = self.format_inputs(scan, beam, ptvs, oars, body)

        # Forward pass
        pred = self.model(*inputs)

        # Format outputs
        pred = self.format_outputs(pred, transform_params)

        # Return prediction
        return pred
    
    def calculate_loss(self, scan, beam, ptvs, oars, body, dose):
        """
        Calculate loss.
        """

        # Compute prior loss
        prior = 0
        n_parameters = 0
        for name, param in self.model.named_parameters():
            n_parameters += param.numel()
            if 'bias' in name:
                prior += (param + .1).pow(2).sum()  # Bias relu threholds at -0.1 to prevent dead neurons
            else:
                prior += param.pow(2).sum()
        prior /= n_parameters

        # Get prediction
        pred = self(scan, beam, ptvs, oars, body)

        # Compute likelihood
        likelihood = F.mse_loss(pred, dose)

        # Add diffusion loss
        if self.architecture.lower() in ["diffunet", "diffvit"]:
            inputs = self.format_inputs(scan, beam, ptvs, oars, body)[0]             # Format inputs
            dose_scale = ptvs.max().item() if ptvs.max() > 0 else 1                  # Get dose scale
            dose_for_diff = resize_image_3d(dose, self.shape)[0]                     # Resize dose
            dose_for_diff = dose_for_diff / dose_scale                               # Rescale dose
            loss_diff = self.model.calculate_diffusion_loss(dose_for_diff, *inputs)  # Calculate diffusion loss
            likelihood += dose_scale * loss_diff                                     # Add diffusion loss (scaled by dose)

        # Compute competition loss
        loss_competition = competition_loss(pred, dose, body)

        # Compute dose volume histogram loss
        loss_dvh = dvh_loss(
            pred, dose, 
            structures=torch.cat([(ptvs!=0), oars, body], dim=1)
        )

        # Compute total loss
        loss = (
            likelihood 
            + prior 
            + loss_competition 
            + loss_dvh
        )

        # Check for NaN and Inf
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError('Loss is NaN or Inf.')

        # Return loss
        return loss



# Example usage
if __name__ == "__main__":
    
    # Initialize inputs
    shape_model = (128, 128, 128)
    shape_data = (99, 205, 188)
    scan = torch.randn(1, 1, *shape_data)
    beam = torch.randn(1, 1, *shape_data)
    ptvs = torch.randn(1, 1, *shape_data)
    oars = torch.randn(1, 1, *shape_data)
    body = torch.randn(1, 1, *shape_data)
    n_channels = scan.shape[1] + beam.shape[1] + ptvs.shape[1] + oars.shape[1] + body.shape[1]
    
    # Initialize model    
    model = DosePredictionModel(
        architecture="test", 
        n_channels=n_channels,
    )

    # Forward pass
    with torch.no_grad():
        pred = model(scan, beam, ptvs, oars, body)
    
    # Done
    print("Done.")

