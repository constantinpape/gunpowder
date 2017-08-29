from .freezable import Freezable
from copy import deepcopy
from gunpowder.coordinate import Coordinate
import logging
import numpy as np

logger = logging.getLogger(__name__)

class VolumeType(Freezable):
    '''Describes general properties of a volume type.

    Args:

        identifier (string):
            A human readable identifier for this volume type. Will be used as a 
            static attribute in :class:`VolumeTypes`. Should be upper case (like 
            ``RAW``, ``GT_LABELS``).
    '''

    def __init__(self, identifier):
        self.identifier = identifier
        self.hash = hash(identifier)
        self.freeze()

    def __eq__(self, other):
        return hasattr(other, 'identifier') and self.identifier == other.identifier

    def __hash__(self):
        return self.hash

    def __repr__(self):
        return self.identifier

class VolumeTypes:
    '''An expandable collection of volume types, which initially contains:

        =================================  ====================================================
        identifier                         purpose
        =================================  ====================================================
        ``RAW``                            Raw intensity volumes.
        ``ALPHA_MASK``                     Alpha mask for blending
                                           raw volumes
                                           (used in :class:`DefectAugment`).
        ``GT_LABELS``                      Ground-truth object IDs.
        ``GT_AFFINITIES``                  Ground-truth affinities.
        ``GT_MASK``                        Binary mask (1-use, 0-don't use) on ground-truth. No 
                                           assumptions about masked out area (i.e., end of 
                                           ground-truth).
        ``GT_IGNORE``                      Binary mask (1-use, 0-don't use) on ground-truth. 
                                           Assumes that transition between 0 and 1 lies on an 
                                           object boundary.
        ``PRED_AFFINITIES``                Predicted affinities.
        ``LOSS_SCALE``                     Used for element-wise multiplication with loss for
                                           training.
        ``LOSS_GRADIENT``                  Gradient of the training loss.
        ``GT_BM_PRESYN``                   Ground truth of binary map for presynaptic locations
        ``GT_BM_PRESYN``                   Ground truth of binary map for postsynaptic locations
        ``GT_MASK_EXCLUSIVEZONE_PRESYN``   ExculsiveZone binary mask (1-use, 
                                           0-don't use) around presyn locations
        ``GT_MASK_EXCLUSIVEZONE_POSTSYN``  ExculsiveZone binary mask (1-use, 
                                           0-don't use) around postsyn locations
        ``PRED_BM_PRESYN``                 Predicted presynaptic locations
        ``PRED_BM_POSTSYN``                Predicted postsynaptic locations
        =================================  ====================================================

    New volume types can be added with :func:`register_volume_type`.
    '''
    pass

def register_volume_type(identifier):
    '''Register a new volume type.

    For example, the following call::

            register_volume_type('IDENTIFIER')

    will create a new volume type available as ``VolumeTypes.IDENTIFIER``. 
    ``VolumeTypes.IDENTIFIER`` can then be used in dictionaries, as it is done 
    in :class:`BatchRequest` and :class:`ProviderSpec`, for example.
    '''
    volume_type = VolumeType(identifier)
    logger.debug("Registering volume type " + str(volume_type))
    setattr(VolumeTypes, volume_type.identifier, volume_type)

register_volume_type('RAW')
register_volume_type('ALPHA_MASK')
register_volume_type('GT_LABELS')
register_volume_type('GT_AFFINITIES')
register_volume_type('GT_MASK')
register_volume_type('GT_IGNORE')
register_volume_type('PRED_AFFINITIES')
register_volume_type('LOSS_SCALE')
register_volume_type('LOSS_GRADIENT')
register_volume_type('MALIS_COMP_LABEL')

register_volume_type('GT_BM_PRESYN')
register_volume_type('GT_BM_POSTSYN')
register_volume_type('GT_MASK_EXCLUSIVEZONE_PRESYN')
register_volume_type('GT_MASK_EXCLUSIVEZONE_POSTSYN')
register_volume_type('PRED_BM_PRESYN')
register_volume_type('PRED_BM_POSTSYN')
register_volume_type('LOSS_GRADIENT_PRESYN')
register_volume_type('LOSS_GRADIENT_POSTSYN')

register_volume_type('LOSS_SCALE_BM_PRESYN')
register_volume_type('LOSS_SCALE_BM_POSTSYN')


class Volume(Freezable):

    def __init__(self, data, spec=None):

        self.spec = deepcopy(spec)
        self.data = data

        if spec is not None:
            for d in range(len(spec.voxel_size)):
                assert spec.voxel_size[d]*data.shape[-spec.roi.dims()+d] == spec.roi.get_shape()[d], \
                        "ROI %s does not align with voxel size %s * data shape %s"%(spec.roi, spec.voxel_size, data.shape)

        self.freeze()

    def crop(self, roi, copy=True):
        '''Create a cropped copy of this Volume.

        Args:

            roi(:class:``Roi``): ROI in world units to crop to.

            copy(bool): Make a copy of the data (default).
        '''

        assert self.spec.roi.contains(roi)

        voxel_size = self.spec.voxel_size
        data_roi = (roi - self.spec.roi.get_offset())/voxel_size
        slices = data_roi.get_bounding_box()

        while len(slices) < len(self.data.shape):
            slices = (slice(None),) + slices

        data = self.data[slices]
        if copy:
            data = np.array(data)

        spec = deepcopy(self.spec)
        spec.roi = deepcopy(roi)
        return Volume(data, spec)
