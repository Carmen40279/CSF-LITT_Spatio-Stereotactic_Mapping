from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv


MRI_PATH = Path(
    "data/sample/sub-01_ses-forrestgump_T1w.nii.gz"
)


def show_mri():
    image = nib.load(str(MRI_PATH))
    data = np.asarray(image.dataobj)

    print("MRI shape:", data.shape)
    print("Voxel spacing:", image.header.get_zooms()[:3])

    grid = pv.ImageData()


    grid.dimensions = data.shape
    grid.spacing = tuple(float(x) for x in image.header.get_zooms()[:3])
    grid.origin = (0.0, 0.0, 0.0)


    grid.point_data["MRI"] = data.flatten(order="F")

    plotter = pv.Plotter()
    plotter.add_volume(
        grid,
        scalars="MRI",
        opacity="sigmoid",
        cmap="gray",
    )

    plotter.add_text(
        "ThermoSEEG — MRI Volume",
        position="upper_left",
        font_size=14,
    )

    plotter.show()


if __name__ == "__main__":
    show_mri()