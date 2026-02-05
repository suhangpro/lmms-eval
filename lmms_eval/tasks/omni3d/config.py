"""
Configuration for Omni3D dataset paths.

Set environment variables to override the default paths:
  - OMNI3D_IMAGE_ROOT: Path to omni3d_images directory
  - OMNI3D_JSON_ROOT: Path to Omni3D JSON annotations directory
"""

import os

# Path to this module's directory (for category_meta.json)
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# Image root directory - can be overridden by environment variable
IMAGE_ROOT = os.environ.get(
    "OMNI3D_IMAGE_ROOT",
    "/home/hangsu/Data/Omni3D/omni3d_images"
)

# JSON annotation root directory - can be overridden by environment variable
JSON_ROOT = os.environ.get(
    "OMNI3D_JSON_ROOT",
    "/home/hangsu/Data/Omni3D/Omni3D_json/datasets/Omni3D"
)

# Category metadata file (shipped with the code)
CATEGORY_META_PATH = os.path.join(MODULE_DIR, "category_meta.json")
