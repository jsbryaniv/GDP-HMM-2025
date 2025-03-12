
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
        if architecture.lower() == "unet":
            # Unet3D
            from architectures.unet import Unet3D
            kwargs = {
                'in_channels': n_channels,
                'out_channels': 1,
                **kwargs,
            }
            self.model = Unet3D(**kwargs)
        elif architecture.lower() == "vit":
            # ViT3D
            from architectures.vit import ViT3D
            kwargs = {
                'in_channels': n_channels,
                'out_channels': 1,
                'shape': shape,
                **kwargs,
            }
            self.model = ViT3D(**kwargs)
        elif architecture.lower() == "crossattnunet":
            # CrossAttnUnetModel
            from architectures.crossattnunet import CrossAttnUnetModel
            kwargs = {
                'in_channels': 4,
                'out_channels': 1,
                'n_cross_channels_list': [1, 4, n_channels-5],  # scan, (beam, ptvs), (oars, body)
                **kwargs,
            }
            self.model = CrossAttnUnetModel(**kwargs)
        elif architecture.lower() == "crossvit":
            # CrossViT3d
            from architectures.crossvit import CrossViT3d
            kwargs = {
                'in_channels': 4,
                'out_channels': 1,
                'n_cross_channels_list': [1, 4, n_channels-5],  # scan, (beam, ptvs), (oars, body)
                'shape': shape,
                **kwargs,
            }
            self.model = CrossViT3d(**kwargs)
        elif architecture.lower() == "convformer":
            # ConvFormer
            from architectures.convformer import ConvformerModel
            kwargs = {
                'in_channels': n_channels,
                'out_channels': 1,
                **kwargs,
            }
            self.model = ConvformerModel(**kwargs)
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
    def from_checkpoint(cls, checkpoint_path):
        """
        Load model from checkpoint.
        """

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        # Initialize model
        model = cls(**checkpoint['model_config'])
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Return model
        return model
        
    def format_inputs(self, scan, beam, ptvs, oars, body):

        # Reshape inputs
        transform_params=None  # Initialize to None
        if self.shape is not None:
            # ### Pad to 256x256x256 ###
            # # Get pad info
            # shape_target = (256, 256, 256)
            # shape_origin = scan.shape[2:]
            # padding = [shape_target[i] - shape_origin[i] for i in range(3)]
            # padding = [(p//2, p-p//2) for p in padding]
            # padding = tuple(sum(padding[::-1], ()))  # Flatten and reverse order
            # # Pad
            # scan = F.pad(scan, padding, value=scan.min())
            # beam = F.pad(beam, padding)
            # ptvs = F.pad(ptvs, padding)
            # oars = F.pad(oars, padding)
            # body = F.pad(body, padding)
            ### Resize to shape ###
            scan, resize_params = resize_image_3d(scan, self.shape, fill_value=scan.min())
            beam, _ = resize_image_3d(beam, self.shape)
            ptvs, _ = resize_image_3d(ptvs, self.shape)
            oars, _ = resize_image_3d(oars, self.shape)
            body, _ = resize_image_3d(body, self.shape)
            ### Update transform params ###
            transform_params = {
                # 'original_shape': shape_origin,
                # 'padding': padding,
                'resize': resize_params,
            }

        # Check architecture
        if self.architecture.lower() in ["unet", "vit", "convformer"]:
            # Concatenate inputs
            x = torch.cat([scan, beam, ptvs, oars, body], dim=1)
            inputs = (x,)
        elif self.architecture.lower() in ["crossattnunet", "crossvit"]:
            # Separate contexts
            x = torch.cat([beam, ptvs], dim=1).clone()
            y_list = [
                scan, 
                torch.cat([beam, ptvs], dim=1), 
                torch.cat([oars, body], dim=1).float(),
            ]
            inputs = (x, y_list)

        # Return inputs
        return inputs, transform_params
    
    def format_outputs(self, x, transform_params):
        
        # Reshape prediction
        if self.shape is not None:
            ### Extract transform params ###
            # original_shape = transform_params['original_shape']
            # padding = transform_params['padding']
            resize_params = transform_params['resize']
            ### Resize to padded shape ###
            x = reverse_resize_3d(x, resize_params)
            ### Unpad to original shape ###
            # x = x[
            #     :, :, 
            #     padding[4]:256-padding[5],
            #     padding[2]:256-padding[3],
            #     padding[0]:256-padding[1],
            # ]

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
        inputs, transform_params = self.format_inputs(scan, beam, ptvs, oars, body)
        pred = self.model(*inputs)
        pred = self.format_outputs(pred, transform_params)

        # Compute likelihood
        likelihood = F.mse_loss(pred, dose)

        # Compute competition loss
        loss_competition = competition_loss(pred, dose, body)

        # Compute dose volume histogram loss
        loss_dvh = dvh_loss(
            pred, dose, 
            structures=torch.cat([(ptvs!=0), oars, body], dim=1)
        )

        # # Compute reconstruction loss
        # if self.architecture.lower() in ['crossattnunet', 'crossvit']:
        #     """Compute reconstruction loss for cross-attention-unet."""
        #     # Get context
        #     y_list = inputs[1]
        #     # Corrupt context
        #     if self.architecture.lower() == 'crossvit':
        #         y_list_corrupt = [block_mask_3d(y, p=0.5) for y in y_list]
        #     else:
        #         y_list_corrupt = y_list
        #     # Get reconstructions
        #     recons = self.model.autoencode_context(y_list_corrupt)
        #     # Compute likelihood loss
        #     likelihood_recon = (
        #         F.mse_loss(y_list[0], recons[0])
        #         + F.mse_loss(y_list[1], recons[1])
        #         + F.binary_cross_entropy_with_logits(recons[2], y_list[2])  # Use BCE for binary masks
        #     )
        #     # Combine losses
        #     likelihood += likelihood_recon

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

        # # Plot
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(2, len(y_list)+1, figsize=(4*len(y_list)+1, 8))
        # for i, (y, y_ae) in enumerate(zip([dose]+y_list_corrupt, [pred]+recons)):
        #     if i >= 3:
        #         y_ae = torch.sigmoid(y_ae)
        #     ax[0, i].imshow(y[0,0,y.shape[2]//2].detach().cpu().numpy())
        #     ax[1, i].imshow(y_ae[0,0,y.shape[2]//2].detach().cpu().numpy())
        #     ax[0, i].set_title(f'({y.min().item():.2f}, {y.max().item():.2f})')
        #     ax[1, i].set_title(f'({y_ae.min().item():.2f}, {y_ae.max().item():.2f})')
        # plt.show()
        # plt.pause(1)
        # plt.savefig('_image.png')
        # plt.close()
        # print(loss.item())

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
        architecture="unet", 
        shape=shape_model, 
        n_channels=n_channels, 
        n_blocks=0,             # Dummy model for testing (0 blocks)
        n_layers_per_block=0,   # Dummy model for testing (0 layers)
        n_features=1,           # Dummy model for testing (1 feature)
        n_groups=1,             # Dummy model for testing (1 group)
    )

    # Forward pass
    with torch.no_grad():
        pred = model(scan, beam, ptvs, oars, body)
    
    # Done
    print("Done.")

