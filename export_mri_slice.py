from pathlib import Path

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Existing MRI file
mri_path = Path("data/sample/sub-01_ses-forrestgump_T1w.nii.gz")

if not mri_path.exists():
    raise FileNotFoundError(
        f"Cannot find MRI file: {mri_path.resolve()}"
    )

# Load the MRI volume
image = nib.load(str(mri_path))
volume = image.get_fdata()

print("MRI dimensions:", volume.shape)
print("Voxel spacing (mm):", image.header.get_zooms()[:3])

# Extract the middle axial slice for an upload-format test.
# This is an MRI slice, NOT a CT scan.
slice_index = volume.shape[2] // 2
slice_data = volume[:, :, slice_index]

# Export as a regular PNG image.
output_path = Path("data/sample/thermoseeg_test_slice.png")

plt.figure(figsize=(8, 8))
plt.imshow(
    np.rot90(slice_data),
    cmap="gray",
    interpolation="nearest",
)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight",
    pad_inches=0,
)
plt.close()

print(f"Exported image: {output_path.resolve()}")