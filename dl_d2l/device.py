import torch
import platform
import subprocess
import sys


def check_amd_gpu():
    """检测系统中是否存在 AMD GPU 并判断是否支持 ROCm"""
    try:
        # 不同系统的 AMD GPU 检测方式
        if platform.system() == "Linux":
            # Linux 下通过 lspci 检测 AMD GPU
            result = subprocess.check_output(
                ["lspci"],
                stderr=subprocess.DEVNULL,
                text=True
            )
            return "AMD" in result and "VGA compatible controller" in result
        elif platform.system() == "Windows":
            # Windows 下通过 dxdiag 简化检测（需要管理员权限）
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
    自动检测并返回最优可用设备
    优先级: NVIDIA GPU > AMD GPU (ROCm) > CPU
    """
    # 1. 检测 NVIDIA GPU (CUDA)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ 检测到 NVIDIA GPU: {gpu_name} (共 {gpu_count} 块)")
        print(f"   CUDA 版本: {torch.version.cuda}")
        return device

    # 2. 检测 AMD GPU (ROCm)
    # ROCm 主要支持 Linux 系统，Windows 下 AMD GPU 目前只能用 CPU 模式或 DirectML
    try:
        # 尝试导入 ROCm 相关模块（Linux）
        if platform.system() == "Linux" and check_amd_gpu():
            # 验证 ROCm 是否安装成功
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                device = torch.device("rocm")
                gpu_count = torch.cuda.device_count()  # ROCm 兼容 CUDA 接口
                print(f"✅ 检测到 AMD GPU (ROCm)")
                print(f"   ROCm 版本: {torch.version.hip}")
                return device
    except:
        pass

    # 3. Windows 下 AMD GPU 降级方案（DirectML）
    if platform.system() == "Windows" and check_amd_gpu():
        try:
            # 安装并导入 DirectML 后端
            import torch_directml
            dml_device = torch_directml.device()
            print(f"✅ 检测到 AMD GPU (Windows DirectML)")
            print(f"   DirectML 设备 ID: {torch_directml.device_count()}")
            return dml_device
        except ImportError:
            print("⚠️ AMD GPU 检测到但未安装 torch-directml，将使用 CPU")
            print("   安装命令: pip install torch-directml")

    # 4. 兜底方案：CPU
    print("⚠️ 未检测到可用 GPU，将使用 CPU")
    return torch.device("cpu")


def test_device_performance(device):
    """测试设备基本运算能力"""
    print("\n=== 设备性能测试 ===")
    # 创建测试张量并移到目标设备
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)

    # 测试矩阵乘法
    import time
    start = time.time()
    for _ in range(100):
        z = torch.matmul(x, y)
    # 确保计算完成（GPU 异步执行）
    if device.type in ["cuda", "rocm"]:
        torch.cuda.synchronize()
    end = time.time()

    print(f"设备类型: {device}")
    print(f"100次矩阵乘法耗时: {end - start:.4f} 秒")
    print(f"张量是否在目标设备: {x.device == device}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 打印系统信息
    print("=== 系统环境检测 ===")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python 版本: {sys.version.split()[0]}")

    # 获取最优设备
    device = get_available_device()

    # 测试设备
    test_device_performance(device)

    # 使用示例：在目标设备上定义模型
    print("\n=== 模型部署示例 ===")
    # 定义简单模型并移到目标设备
    model = torch.nn.Linear(10, 1).to(device)
    # 生成输入数据并移到目标设备
    input_data = torch.randn(5, 10).to(device)
    # 前向传播
    output = model(input_data)
    print(f"模型输入设备: {input_data.device}")
    print(f"模型参数设备: {next(model.parameters()).device}")
    print(f"输出结果形状: {output.shape}")


