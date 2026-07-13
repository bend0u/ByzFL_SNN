import torch
import torch.nn as nn
from models.cnn_mnist_snn import get_model

def measure_gradient_sparsity(is_snn=True, surrogate='tri', beta=2.0, alpha=1.2):
    # Setup model kwargs
    model_kwargs = {
        'is_snn': is_snn,
        'dataset_name': 'mnist',
        'nb_labels': 10,
        'encoding': {'type': 'constant', 'time_steps': 10},
        'model_params': {'beta': 0.95}
    }
    
    if is_snn:
        model_kwargs['model_params']['surrogate_gradient'] = surrogate
        if surrogate == 'tri':
            model_kwargs['model_params']['surrogate_params'] = {'beta': beta}
        else:
            model_kwargs['model_params']['surrogate_params'] = {'alpha': alpha}

    # Initialize model
    model = get_model(**model_kwargs)
    
    # Create dummy data (Batch size 128, MNIST: 1 channel, 28x28)
    dummy_input = torch.randn(128, 1, 28, 28)
    dummy_target = torch.randint(0, 10, (128,))
    
    criterion = nn.CrossEntropyLoss()
    
    # Forward pass
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    
    # Backward pass
    loss.backward()
    
    # Calculate sparsity
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # Flatten gradient and count zeros
            grad = param.grad.view(-1)
            total = grad.numel()
            # We consider a gradient value exactly 0 as sparse
            zeros = (grad == 0.0).sum().item()
            
            total_params += total
            zero_params += zeros
            
    sparsity = zero_params / total_params if total_params > 0 else 0
    p = 1.0 - sparsity
    
    model_type = f"SNN ({surrogate})" if is_snn else "CNN (Dense)"
    print(f"[{model_type}] Total params: {total_params}, Zero gradients: {zero_params}")
    print(f"[{model_type}] Sparsity: {sparsity*100:.2f}% (zeros)")
    print(f"[{model_type}] Non-zero rate (p): {p*100:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    measure_gradient_sparsity(is_snn=False) # CNN
    measure_gradient_sparsity(is_snn=True, surrogate='tri') # SNN Tri
    measure_gradient_sparsity(is_snn=True, surrogate='atan') # SNN Atan
