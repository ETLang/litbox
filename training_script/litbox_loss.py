import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

class HdrLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.5, base_weight=1.0):
        """
        Args:
            alpha: Hyperparameter for the adaptive weighting. Higher values
                   give more weight to brighter pixels. A value of 1.0 or 2.0
                   is a good starting point.
            beta: Weighting for the gradient loss term.
            gamma: Weighting for the standard L1 loss term.
            base_weight: A constant added to the linear-space ground truth
                         before adaptive weighting. This prevents low-luminance
                         pixels from having a near-zero weight.
        """
        super(HdrLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.base_weight = base_weight
        self.l1_loss = nn.L1Loss()
        
        # Sobel filters for gradient calculation
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, pred_linear, target_linear):
        # Ensure pred and target have the same shape and move filters to device
        if pred_linear.shape != target_linear.shape:
            raise ValueError("Input tensors must have the same shape.")
        
        device = pred_linear.device
        sobel_x_dev = self.sobel_x.to(device)
        sobel_y_dev = self.sobel_y.to(device)

        # --- 1. Adaptive L2 Loss (Primary Loss) ---
        # The weight is based directly on the linear luminance values, but with a
        # base_weight added to ensure even dark pixels have a non-zero contribution.
        weights = (target_linear + self.base_weight) ** self.alpha
        
        # Calculate the L2 difference in linear space
        l2_diff = (pred_linear - target_linear) ** 2
        
        # Apply the weights to the L2 difference
        adaptive_l2_loss = torch.mean(weights * l2_diff)

        # --- 2. Gradient Loss ---
        # This term helps preserve edges and fine details.
        
        # Expand the filters to match the number of input channels (C)
        sobel_x_dev = sobel_x_dev.repeat(pred_linear.shape[1], 1, 1, 1)
        sobel_y_dev = sobel_y_dev.repeat(pred_linear.shape[1], 1, 1, 1)
        
        # Calculate gradients using grouped convolutions
        grad_pred_x = F.conv2d(pred_linear, sobel_x_dev, padding='same', groups=pred_linear.shape[1])
        grad_pred_y = F.conv2d(pred_linear, sobel_y_dev, padding='same', groups=pred_linear.shape[1])
        grad_target_x = F.conv2d(target_linear, sobel_x_dev, padding='same', groups=target_linear.shape[1])
        grad_target_y = F.conv2d(target_linear, sobel_y_dev, padding='same', groups=target_linear.shape[1])

        # Calculate L1 difference between gradients
        gradient_loss = self.l1_loss(grad_pred_x, grad_target_x) + self.l1_loss(grad_pred_y, grad_target_y)

        # --- 3. Standard L1 Loss (for additional stability) ---
        # We include a standard L1 loss as a base to ensure general convergence
        l1_loss = self.l1_loss(pred_linear, target_linear)

        # Combine all loss terms
        total_loss = adaptive_l2_loss + (self.beta * gradient_loss) + (self.gamma * l1_loss)
        
        return total_loss


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device, requires_grad=False, feature_layers=['relu3_3', 'relu4_3']):
        super(VGGPerceptualLoss, self).__init__()
        
        # Load pre-trained VGG16 model
        vgg16_features = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.to(device)
        
        # Mapping from common VGG layer names to their nn.Sequential indices
        # These indices are specific to torchvision.models.vgg16.features
        self.vgg_layer_indices = {
            'relu1_1': 0,  # conv1_1
            'relu1_2': 2,  # conv1_2
            'relu2_1': 5,  # conv2_1
            'relu2_2': 7,  # conv2_2
            'relu3_1': 10, # conv3_1
            'relu3_2': 12, # conv3_2
            'relu3_3': 14, # conv3_3
            'relu4_1': 17, # conv4_1
            'relu4_2': 19, # conv4_2
            'relu4_3': 21, # conv4_3
            'relu5_1': 24, # conv5_1
            'relu5_2': 26, # conv5_2
            'relu5_3': 28  # conv5_3
        }

        # Select only the VGG layers needed for feature extraction
        self.feature_layers_to_extract = sorted([self.vgg_layer_indices[name] for name in feature_layers])
        
        # Build the VGG model up to the maximum required layer index
        self.vgg_model = nn.Sequential()
        for i, layer in enumerate(vgg16_features):
            self.vgg_model.add_module(str(i), layer)
            if i >= max(self.feature_layers_to_extract):
                break
        
        # Freeze VGG parameters to prevent them from being updated during training
        if not requires_grad:
            for param in self.vgg_model.parameters():
                param.requires_grad = False
        
        # Set VGG to evaluation mode
        self.vgg_model.eval()

        # ImageNet normalization parameters (VGG expects 0-1 then normalized)
        # You might need to adjust this if your model outputs are in a different range (e.g., -1 to 1)
        # and adjust the input to VGG accordingly.
        self.normalize_input = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])

    def forward(self, input_image, target_image):
        # Normalize images for VGG input
        # Only normalize when using PNGs. Not EXR        
        # normalized_input = self.normalize_input(input_image)
        # normalized_target = self.normalize_input(target_image)
        normalized_input = input_image
        normalized_target = target_image

        features_input = []
        features_target = []
        
        x_in = normalized_input
        x_target = normalized_target
        
        for i, layer in enumerate(self.vgg_model):
            x_in = layer(x_in)
            x_target = layer(x_target)
            
            if i in self.feature_layers_to_extract:
                features_input.append(x_in)
                features_target.append(x_target)
        
        perceptual_loss = 0
        for f_in, f_target in zip(features_input, features_target):
            # Use MSE loss between feature maps
            perceptual_loss += F.mse_loss(f_in, f_target)
        
        return perceptual_loss

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size)

    def create_window(self, window_size):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(1, 1, window_size, window_size).contiguous()
        return window

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                            for x in range(window_size)])
        return gauss/gauss.sum()

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = self.window.to(img1.device)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
