import os
import cv2
import h5py
import numpy as np
from shutil import copyfile

model_path = '2021-06-01--09-01-37_cvppp_affs_standard'
model_id = 'affs_25500'

in_path = os.path.join('../inference', model_path, 'test', model_id, 'seg.hdf')
out_path = os.path.join('../inference', model_path, 'test', model_id, 'submission.h5')
f_seg = h5py.File(in_path, 'r')
seg = f_seg['main'][:]
f_seg.close()

seg = seg[:, 7:-7, 22:-22]
seg = seg.astype(np.uint8)
print(seg.shape, seg.dtype)


example_path = '../data/A1/submission_example.h5'
copyfile(example_path, out_path)
fi = ['plant003','plant004','plant009','plant014','plant019','plant023','plant025','plant028','plant034',
      'plant041','plant056','plant066','plant074','plant075','plant081','plant087','plant093','plant095',
      'plant097','plant103','plant111','plant112','plant117','plant122','plant125','plant131','plant136',
      'plant140','plant150','plant155','plant157','plant158','plant160']
f_out = h5py.File(out_path, 'r+')
for k, fn in enumerate(fi):
    data = f_out['A1']
    img = data[fn]['label'][:]
    del data[fn]['label']
    data[fn]['label'] = seg[k]
f_out.close()
