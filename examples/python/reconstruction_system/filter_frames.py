import os
import argparse
from glob import glob


def main():
    parser = argparse.ArgumentParser(description="Filter out the frames")
    parser.add_argument("--config",
                        help="path to the config file, e.g. ./kinetic/data/filename/config.json",
                        required=True)
    args = parser.parse_args()

    base_path = os.path.dirname(args.config)
    img_paths = sorted(glob(f"{base_path}/color/*.jpg"))
    img_removed_paths = img_paths[1::2]

    for img_removed_path in img_removed_paths:
        depth_removed_path = os.path.join(base_path, f"depth/{os.path.basename(img_removed_path).replace('.jpg', '.png')}") 
        os.remove(img_removed_path)
        os.remove(depth_removed_path)
    
    print(f"Filtering result: {len(img_paths)} -> {len(img_paths)-len(img_removed_paths)}")


if __name__ == "__main__":
    main()