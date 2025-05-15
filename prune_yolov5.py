import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")

def load_model(model_path):
    print(f"Loading model from {model_path}")
    try:
        # Try with weights_only=False (less secure but necessary for some models)
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        print("Model loaded with weights_only=False")
    except Exception as e:
        print(f"Loading failed: {e}")
        raise RuntimeError(f"Failed to load model from {model_path}")
    
    if 'ema' in ckpt and ckpt['ema'] is not None:
        print("Using EMA model")
        model = ckpt['ema'].float()
    else:
        print("Using regular model")
        model = ckpt['model'].float()
    
    model.eval()
    return model, ckpt

def calculate_sparsity(model):
    """Calculate the percentage of zero weights in the model"""
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += torch.sum(param == 0).item()
    
    sparsity = 100.0 * zero_params / total_params if total_params > 0 else 0
    return total_params, zero_params, sparsity

def direct_magnitude_pruning(model, pruning_ratio=0.2):
    """Directly prune weights based on magnitude across the entire model"""
    print(f"Applying direct magnitude pruning with ratio {pruning_ratio}")
    
    # Collect all weights in the model
    all_weights = []
    weight_tensors = []
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_tensors.append((name, param))
            all_weights.append(param.data.abs().flatten())
    
    if not all_weights:
        print("No weights found in the model!")
        return model
    
    # Concatenate all weights and determine threshold
    all_weights = torch.cat(all_weights)
    threshold_index = int(all_weights.numel() * pruning_ratio)
    
    if threshold_index >= all_weights.numel():
        print(f"Warning: Pruning ratio too high, would remove all weights")
        threshold_index = all_weights.numel() - 1
        
    # Sort weights and find threshold value
    sorted_weights, _ = torch.sort(all_weights)
    threshold = sorted_weights[threshold_index]
    
    print(f"Pruning threshold: {threshold:.6f}")
    
    # Apply pruning
    total_weights = 0
    total_pruned = 0
    
    for name, param in weight_tensors:
        mask = param.data.abs() > threshold
        param.data = param.data * mask
        
        weights_count = param.numel()
        pruned_count = weights_count - mask.sum().item()
        
        total_weights += weights_count
        total_pruned += pruned_count
        
        layer_sparsity = 100.0 * pruned_count / weights_count
        print(f"Layer {name}: pruned {pruned_count}/{weights_count} weights ({layer_sparsity:.2f}% sparse)")
    
    overall_sparsity = 100.0 * total_pruned / total_weights if total_weights > 0 else 0
    print(f"Overall: pruned {total_pruned}/{total_weights} weights ({overall_sparsity:.2f}% sparse)")
    
    return model

def visualize_weights(model, save_path):
    """Create a visualization of model weights distribution"""
    weights = []
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.flatten().cpu().numpy()
            weights.append(tensor)
    
    if weights:
        all_weights = np.concatenate(weights)
        
        plt.figure(figsize=(10, 6))
        plt.hist(all_weights, bins=100, alpha=0.7)
        plt.title('Distribution of Weights in YOLOv5 Model')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.yscale('log')  # Log scale to better visualize the zeros
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        print(f"Weight visualization saved to {save_path}")

def save_pruned_model(model, original_ckpt, save_path, sparsity):
    """Save the pruned model in YOLOv5 format"""
    pruned_ckpt = {}
    
    # Copy important keys
    for key in original_ckpt:
        if key != 'model' and key != 'ema':
            pruned_ckpt[key] = original_ckpt[key]
    
    # Update with pruned model
    pruned_ckpt['model'] = model
    
    if 'ema' in original_ckpt and original_ckpt['ema'] is not None:
        pruned_ckpt['ema'] = model  # Use the same pruned model for EMA
    
    # Add pruning metadata
    pruned_ckpt['pruning_info'] = {
        'sparsity': sparsity,
        'method': 'Direct magnitude pruning'
    }
    
    # Save model
    print(f"Saving pruned model to {save_path}")
    torch.save(pruned_ckpt, save_path, _use_new_zipfile_serialization=False)
    print(f"Model saved successfully")

def main():
    model_path = '/mnt/bst/hxu10/hxu10/chanti/yolov5/runs/train/exp/weights/best.pt'
    output_dir = '/mnt/bst/hxu10/hxu10/chanti/yolov5/runs/train/exp/weights/'
    visualization_dir = '/mnt/bst/hxu10/hxu10/chanti/dataset/images/test/'
    
    # Define pruning ratios to try
    pruning_ratios = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    try:
        # Load the model
        model, original_ckpt = load_model(model_path)
        
        # Calculate original parameters
        total_params, zero_params, original_sparsity = calculate_sparsity(model)
        print(f"Original model - Total parameters: {total_params}, Zero parameters: {zero_params}, Sparsity: {original_sparsity:.2f}%")
        
        # Process each pruning ratio
        for ratio in pruning_ratios:
            print(f"\n=== Processing pruning ratio: {ratio:.2f} ===")
            
            # Create output paths
            pruned_save_path = os.path.join(output_dir, f"direct_pruned_{int(ratio*100)}.pt")
            vis_path = os.path.join(visualization_dir, f"direct_pruned_{int(ratio*100)}.png")
            
            try:
                # Create a fresh copy of the model for each ratio
                model_copy = deepcopy(model)
                
                # Apply pruning
                pruned_model = direct_magnitude_pruning(model_copy, pruning_ratio=ratio)
                
                # Calculate sparsity
                total, zeros, sparsity = calculate_sparsity(pruned_model)
                print(f"After pruning - Total parameters: {total}, Zero parameters: {zeros}, Sparsity: {sparsity:.2f}%")
                
                # Visualize weights
                visualize_weights(pruned_model, vis_path)
                
                # Save the pruned model
                save_pruned_model(pruned_model, original_ckpt, pruned_save_path, sparsity)
                
                print(f"Completed processing for ratio {ratio:.2f}")
                
            except Exception as e:
                print(f"Error processing ratio {ratio:.2f}: {str(e)}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()