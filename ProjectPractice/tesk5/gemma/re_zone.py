import os

def remove_zone_identifier(path="."):
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith("Zone.Identifier"):
                full = os.path.join(root, name)
                try:
                    os.remove(full)
                    print(f"Cleaned: {full}")
                except Exception as e:
                    print(f"Failed: {full} - {e}")

if __name__ == "__main__":
    remove_zone_identifier(
        "../../../../../../weChat/xwechat_files/wxid_k9d60hi3dnm112_8a95/msg/file/2025-11/tourtools_lp")
