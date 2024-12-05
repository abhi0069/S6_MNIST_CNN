import torch
import torch.nn as nn
import sys
from model import Net

def test_parameter_count():
    model = Net()
    model.load_state_dict(torch.load('mnist_model.pth', map_location=torch.device('cpu')))
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has {total_params} parameters, which exceeds the limit of 20,000"
    print(f"✓ Parameter count test passed: {total_params} parameters")

def test_batch_norm():
    model = Net()
    model.load_state_dict(torch.load('mnist_model.pth', map_location=torch.device('cpu')))
    has_batch_norm = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert has_batch_norm, "Model does not use Batch Normalization"
    print("✓ Batch Normalization test passed")

def test_dropout():
    model = Net()
    model.load_state_dict(torch.load('mnist_model.pth', map_location=torch.device('cpu')))
    has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
    assert has_dropout, "Model does not use Dropout"
    print("✓ Dropout test passed")

def test_architecture_type():
    model = Net()
    model.load_state_dict(torch.load('mnist_model.pth', map_location=torch.device('cpu')))
    
    # Check for FC layer
    has_fc = any(isinstance(m, nn.Linear) for m in model.modules())
    
    # Check for GAP
    has_gap = any(isinstance(m, nn.AdaptiveAvgPool2d) for m in model.modules())
    
    # Model must use either FC layer or GAP
    assert has_fc or has_gap, "Model must use either Fully Connected layer or Global Average Pooling"
    
    if has_fc:
        print("✓ Architecture test passed: Model uses Fully Connected layer")
    elif has_gap:
        print("✓ Architecture test passed: Model uses Global Average Pooling")

def run_all_tests():
    try:
        test_parameter_count()
        test_batch_norm()
        test_dropout()
        test_architecture_type()
        print("\nAll tests passed! ✨")
        sys.exit(0)
    except AssertionError as e:
        print(f"\nTest failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests() 