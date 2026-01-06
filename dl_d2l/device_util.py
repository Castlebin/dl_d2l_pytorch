import torch
import platform
import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def check_amd_gpu():
    """Check if AMD GPU exists and supports ROCm"""
    try:
        if platform.system() == "Linux":
            result = subprocess.check_output(
                ["lspci"],
                stderr=subprocess.DEVNULL,
                text=True
            )
            return "AMD" in result and "VGA compatible controller" in result
        elif platform.system() == "Windows":
            try:
                result = subprocess.check_output(
                    ["dxdiag", "/t", "dxdiag.txt"],
                    stderr=subprocess.DEVNULL,
                    text=True
                )
                with open("dxdiag.txt", "r") as f:
                    content = f.read()
                return "AMD" in content or "Radeon" in content
            except:
                return False
        else:
            return False
    except:
        return False


def get_available_device():
    """
    Automatically detect and return the best available device.
    Priority: NVIDIA GPU > AMD GPU (ROCm) > CPU
    """
    # 1. Check for NVIDIA GPU (CUDA)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"Detected NVIDIA GPU: {gpu_name} (Total: {gpu_count})")
        logging.info(f"CUDA Version: {torch.version.cuda}")
        return device

    # 2. Check for AMD GPU (ROCm)
    try:
        if platform.system() == "Linux" and check_amd_gpu():
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                device = torch.device("rocm")
                logging.info(f"Detected AMD GPU (ROCm) ")
                logging.info(f"ROCm Version: {torch.version.hip}")
                return device
    except:
        pass

    # 3. Windows AMD GPU fallback (DirectML)
    if platform.system() == "Windows" and check_amd_gpu():
        try:
            import torch_directml
            dml_device = torch_directml.device()
            logging.info("Detected AMD GPU (Windows DirectML)")
            logging.info(f"DirectML Device ID: {torch_directml.device_count()}")
            return dml_device
        except ImportError:
            logging.warning("AMD GPU detected but torch-directml is not installed. Falling back to CPU.")
            logging.warning("Install command: pip install torch-directml")

    # 4. Check for Apple MPS (Metal Performance Shaders)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Detected Apple GPU (MPS)")
        return device

    # 5. Fallback: CPU
    logging.warning("No available GPU detected. Falling back to CPU.")
    return torch.device("cpu")


def test_device_performance(device):
    """Test basic computation performance of the device"""
    logging.info("=== Device Performance Test ===")
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)

    import time
    start = time.time()
    for _ in range(100):
        z = torch.matmul(x, y)
    if device.type in ["cuda", "rocm"]:
        torch.cuda.synchronize()
    end = time.time()

    logging.info(f"Device Type: {device}")
    logging.info(f"Time for 100 matrix multiplications: {end - start:.4f} seconds")
    logging.info(f"Tensors on target device type: {x.device.type == device.type}")


# ==================== Main Program ====================
if __name__ == "__main__":
    logging.info("=== System Environment Detection ===")
    logging.info(f"PyTorch Version: {torch.__version__}")
    logging.info(f"Operating System: {platform.system()} {platform.release()}")
    logging.info(f"Python Version: {sys.version.split()[0]}")

    device = get_available_device()
    test_device_performance(device)

    logging.info("=== Model Deployment Example ===")
    model = torch.nn.Linear(10, 1).to(device)
    input_data = torch.randn(5, 10).to(device)
    output = model(input_data)
    logging.info(f"Model Input Device: {input_data.device}")
    logging.info(f"Model Parameter Device: {next(model.parameters()).device}")
    logging.info(f"Output Shape: {output.shape}")


