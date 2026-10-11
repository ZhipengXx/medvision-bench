"""Lossless ZYX-to-XYZ export; never invent physical orientation or millimetres."""
from pathlib import Path
import nibabel as nib
import numpy as np


def export_array(array_zyx, path, spacing_zyx=(1,1,1), physical_affine_xyz=None):
    xyz = np.ascontiguousarray(array_zyx.transpose(2,1,0))
    affine = physical_affine_xyz if physical_affine_xyz is not None else np.diag([*spacing_zyx[::-1],1.])
    nii = nib.Nifti1Image(xyz, affine)
    if physical_affine_xyz is None:
        # Consistent voxel display transform, unknown anatomical/physical frame.
        nii.header.set_xyzt_units('unknown')
        nii.header['descrip'] = b'Voxel-coordinate display; anatomical orientation/physical units unknown'
        nii.set_qform(affine, code=0)
        nii.set_sform(affine, code=0)
    nib.save(nii, str(path))
    readback = nib.load(str(path))
    rebuilt = np.asanyarray(readback.dataobj).transpose(2,1,0)
    if not np.array_equal(array_zyx, rebuilt):
        raise ValueError('NIfTI round-trip changed values or geometry')
    return dict(shape_zyx=list(array_zyx.shape), shape_xyz=list(xyz.shape),
                voxel_to_output_permutation=[2,1,0], affine=readback.affine.tolist(),
                physical_coordinates_known=physical_affine_xyz is not None,
                coordinate_note='No anatomical orientation inferred' if physical_affine_xyz is None else 'Supplied physical affine',
                roundtrip_exact=True)
