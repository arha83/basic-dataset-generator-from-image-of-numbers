# basic-dataset-generator-from-image-of-numbers
generating .npy dataset and labels out of given image, containing numbers from 0 to 9, using opencv

## input image
![input](https://github.com/arha83/basic-dataset-generator-from-image-of-numbers/blob/master/numbers.png)

## output image
![output](https://github.com/arha83/basic-dataset-generator-from-image-of-numbers/blob/master/output.png)

## output files
it generates _dataset.npy_ and _labels.npy_ in main directory which can be loaded with:
``` python
import numpy as np

dataset= np.load(NPY_DATASET_LOCATION)
labels= np.load(NPY_LABELS_LOCATION)
```
