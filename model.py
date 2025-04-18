
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import custom libraries
from losses import competition_loss, dvh_loss, structure_dose_loss, gradient_loss
from utils import resize_image_3d, reverse_resize_3d, norm_d97


# Create dose prediction model
class DosePredictionModel(nn.Module):
    """Dose prediction model."""
    def __init__(self, architecture, n_channels, shape=None, scale_dose=False, **kwargs):
        super(DosePredictionModel, self).__init__()

        # Check inputs
        if isinstance(shape, int):
            # Convert shape to tuple
            shape = (shape, shape, shape)  
        
        # Set attributes
        self.architecture = architecture
        self.n_channels = n_channels
        self.shape = shape
        self.scale_dose = scale_dose
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
                'n_cross_channels_list': [5, n_channels-2],  # (scan, beam, ptvs), (ptvs, oars, body)
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
        elif architecture.lower() == "moecrossunetlight":
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
                'n_cross_channels_list': [5, n_channels-2],  # (scan, beam, ptvs), (ptvs, oars, body)
                'shape': shape,
                **kwargs,
            }
            self.model = CrossViT3d(**kwargs)
        elif architecture.lower() == "diffunet":
            # DiffUnet3d
            from architectures.diffunet import DiffUnet3d
            kwargs = {
                'in_channels': 1,
                'n_cross_channels_list': [5, n_channels-2],  # (scan, beam, ptvs), (ptvs, oars, body)
                **kwargs,
            }
            self.model = DiffUnet3d(**kwargs)
        elif architecture.lower() == "diffvit":
            # DiffViT3d
            from architectures.diffvit import DiffViT3d
            kwargs = {
                'in_channels': 1,
                'n_cross_channels_list': [5, n_channels-2],  # (scan, beam, ptvs), (ptvs, oars, body)
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
            'scale_dose': self.scale_dose,
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
        
    def format_inputs(self, scan, beam, ptvs, oars, body, dose=None):

        # Initialize transform params
        transform_params = {
            'original_shape': scan.shape[2:],
        }

        # Rescale doses
        if self.scale_dose:
            dose_scale = ptvs.max().item() if ptvs.max() != 0 else 1
            ptvs = ptvs / dose_scale
            if dose is not None:
                dose = dose / dose_scale
            transform_params['dose_scale'] = dose_scale

        # Reshape inputs
        if self.shape is not None:
            ### Resize to shape ###
            scan, resize_params = resize_image_3d(scan, self.shape, fill_value=scan.min())
            beam, _ = resize_image_3d(beam, self.shape)
            ptvs, _ = resize_image_3d(ptvs, self.shape)
            oars, _ = resize_image_3d(oars, self.shape)
            body, _ = resize_image_3d(body, self.shape)
            if dose is not None:
                dose, _ = resize_image_3d(dose, self.shape)
            ### Update transform params ###
            transform_params['type'] = 'resize'
            transform_params['resize'] = resize_params

        # Check architecture
        if self.architecture.lower() in ["test", "unet", "vit", "moeunet", "moevit"]:
            # Concatenate inputs
            inputs = (
                torch.cat([scan, beam, ptvs, oars, body], dim=1),  # All inputs concatenated
            )
        elif self.architecture.lower() in ["crossunet", "crossvit", "moecrossunet", "moecrossvit"]:
            # Separate contexts
            inputs = (
                torch.cat([scan, beam, ptvs, oars, body], dim=1),  # Main input
                torch.cat([scan, beam, ptvs], dim=1),              # Context 1
                torch.cat([ptvs, oars, body], dim=1),              # Context 2
            )
        elif self.architecture.lower() in ["crossunetlight", "crossvitlight", "moecrossunetlight", "moecrossvitlight"]:
            # Separate contexts
            inputs = (
                torch.cat([scan, beam, ptvs, oars, body], dim=1),  # Main input
                torch.cat([scan, beam, ptvs, oars, body], dim=1),  # Context
            )
        elif self.architecture.lower() in ["diffunet", "diffvit"]:
            # Separate contexts
            inputs = (
                torch.cat([scan, beam, ptvs], dim=1),  # Context 1
                torch.cat([ptvs, oars, body], dim=1),  # Context 2
            )
        elif self.architecture.lower() in ["diffunetlight", "diffvitlight"]:
            # Separate contexts
            inputs = (
                torch.cat([scan, beam, ptvs, oars, body], dim=1),  # All context grouped together
            )

        # Convert to float and clone
        inputs = (x.float().clone() for x in inputs)

        # Add dose if available
        if dose is not None:
            inputs = (*inputs, dose)

        # Return inputs
        return inputs, transform_params
    
    def format_outputs(self, x, transform_params):
        
        # Reshape prediction
        if self.shape is not None:
            resize_params = transform_params['resize']
            x = reverse_resize_3d(x, resize_params)

        # Rescale prediction
        if self.scale_dose:
            dose_scale = transform_params['dose_scale']
            x = x * dose_scale

        # Return prediction
        return x
        
    def forward(self, scan, beam, ptvs, oars, body, dose=None, d97=False, return_loss=False):
        
        # Format inputs
        inputs, transform_params = self.format_inputs(scan, beam, ptvs, oars, body, dose=dose)

        # Forward pass
        if return_loss:
            inputs, target = inputs[:-1], inputs[-1]
            pred, loss = self.model(*inputs, target=target, return_loss=True)
        else:
            pred = self.model(*inputs)

        # Format outputs
        pred = self.format_outputs(pred, transform_params)

        # Normalize prediction
        if d97:
            pred = pred * body
            pred = norm_d97(pred, ptvs)

        # Return prediction
        if return_loss:
            return pred, loss
        else:
            return pred
    
    def calculate_loss(self, scan, beam, ptvs, oars, body, dose):
        """
        Calculate loss.
        """

        # Forward pass
        if self.architecture.lower() in ["diffunet", "diffvit", "diffunetlight", "diffvitlight"]:
            pred, loss_diffusion = self(scan, beam, ptvs, oars, body, dose=dose, return_loss=True)
        else:
            pred = self(scan, beam, ptvs, oars, body)
            loss_diffusion = 0

        # Compute likelihood
        likelihood = F.mse_loss(pred, dose)

        # Compute prior loss
        prior = 0
        n_parameters = 0
        for name, param in self.model.named_parameters():
            n_parameters += param.numel()
            if 'bias' in name:
                prior += (param + .1).pow(2).sum()  # Bias relu threholds at -0.1 to prevent dead neurons
            else:
                prior += param.pow(2).sum()
        prior = prior / n_parameters

        # Compute competition loss
        loss_competition = competition_loss(pred, dose, body)

        # Compute dose volume histogram loss
        loss_dvh = dvh_loss(
            pred, dose, 
            structures=torch.cat([(ptvs!=0), oars, body], dim=1)
        )

        # # Compute structure dose loss
        # loss_structure = structure_dose_loss(
        #     pred, dose, 
        #     structures=torch.cat([(ptvs!=0), oars, body], dim=1)
        # )

        # # Compute gradient loss
        # loss_gradient = gradient_loss(pred, dose)

        # import matplotlib.pyplot as plt
        # from plotting import plot_images
        # pred_d97 = self(scan, beam, ptvs, oars, body, d97=True)
        # plot_images(dose=dose[0, :, 64], pred=pred[0, :, 64], pred_d97=pred_d97[0, :, 64])
        # plt.savefig('_image.png', dpi=900)

        # Compute total loss
        loss = (
            likelihood 
            + prior 
            + loss_competition 
            + loss_dvh
            # + loss_structure
            # + loss_gradient
            + loss_diffusion
        )

        # Check for NaN and Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Nans or Infs in Loss!!!")
            print(f"--NaNs in Prediction: {torch.isnan(pred).sum()}")
            print(f"--Infs in Prediction: {torch.isinf(pred).sum()}")
            print(f"--NaNs in Dose:       {torch.isnan(dose).sum()}")
            print(f"--Infs in Dose:       {torch.isinf(dose).sum()}")
            print(f"--Loss MSE =          {F.mse_loss(pred, dose)}")
            print(f"--Loss Likelihood =   {likelihood}")
            print(f"--Loss Prior =        {prior}")
            print(f"--Loss Competition =  {loss_competition}")
            print(f"--Loss DVH =          {loss_dvh}")
            # print(f"--Loss Structure =    {loss_structure}")
            # print(f"--Loss Gradient =      {loss_gradient}")
            print(f"--Loss Diffusion =    {loss_diffusion}")
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

