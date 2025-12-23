def enable_matplotlib_chinese(verbose=True):
    """
    100% 可靠启用 matplotlib 中文 & 负号显示
    适用于 Google Colab / Linux（matplotlib 3.6+）
    """

    import sys
    if not sys.platform.startswith("linux"):
        print(f'当前平台 "{sys.platform}" 不适用此方法')
        return

    import os
    import subprocess
    from matplotlib import rcParams, font_manager

    def log(msg):
        if verbose:
            print(f"[matplotlib-cn] {msg}")

    # 1️⃣ 确保字体包存在
    log("检查中文字体 …")
    subprocess.run(
        ["apt-get", "-qq", "install", "-y", "fonts-noto-cjk"],
        check=True
    )

    # 2️⃣ 在系统字体目录中搜索 CJK 字体文件
    font_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
    ]

    candidates = []
    for font_dir in font_dirs:
        for root, _, files in os.walk(font_dir):
            for f in files:
                name = f.lower()
                if (
                        ("notosanscjk" in name or "notoserifcjk" in name)
                        and (name.endswith(".ttc") or name.endswith(".otf"))
                ):
                    candidates.append(os.path.join(root, f))

    if not candidates:
        raise RuntimeError("未能找到任何 Noto CJK 字体文件")

    font_path = candidates[0]
    log(f"使用字体文件: {font_path}")

    # 3️⃣ 强制注册字体（绕过 font cache）
    font_manager.fontManager.addfont(font_path)
    font_prop = font_manager.FontProperties(fname=font_path)
    font_name = font_prop.get_name()

    # 4️⃣ 设置 matplotlib
    rcParams["font.sans-serif"] = [font_name]
    rcParams["axes.unicode_minus"] = False

    log(f"matplotlib 已启用中文字体: {font_name} ✅")

