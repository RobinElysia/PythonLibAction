"""图片处理工具函数"""
import base64
import os


def image_file_to_base64(image_path):
    """图片路径转base64"""
    if not image_path:
        return None, None
    
    # 尝试多个可能的路径
    paths_to_try = [
        image_path,
        os.path.join(os.getcwd(), image_path.lstrip('/')),
        f"data/images/{os.path.basename(image_path)}"
    ]
    
    actual_path = None
    for path in paths_to_try:
        if os.path.exists(path):
            actual_path = path
            break
    
    if not actual_path:
        return None, None

    try:
        with open(actual_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode('utf-8')

        ext = os.path.splitext(actual_path)[1].lower().lstrip('.')

        mime_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'bmp': 'image/bmp',
            'webp': 'image/webp',
            'tiff': 'image/tiff',
            'tif': 'image/tiff',
            'svg': 'image/svg+xml',
            'ico': 'image/x-icon',
        }

        if ext not in mime_types:
            return None, None

        mime_type = mime_types.get(ext, "image/jpeg")
        return base64_data, mime_type

    except Exception:
        return None, None
