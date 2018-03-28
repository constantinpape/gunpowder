import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.array import Array

from multiscale_affinities import compute_multiscale_affinities

logger = logging.getLogger(__name__)


class AddMultiscaleAffinities(BatchFilter):
    '''Add an array with affinities for a given label array and downsampled with
    the given block shape to the batch.

    Args:

        block_shape(list of ints): List of factors for downsampling.

        labels(:class:``ArrayKey``): The array to read the labels from.

        affinities(:class:``ArrayKey``): The array to generate containing
            the affinities.

        labels_mask(:class:``ArrayKey``, optional): The array to use as a
            mask for ``labels``.

        affinities_mask(:class:``ArrayKey``, optional): The array to
            generate containing the affinitiy mask, as derived from parameter
            ``labels_mask``.
    '''

    def __init__(
            self,
            block_shape,
            labels,
            affinities,
            labels_mask=None,
            affinities_mask=None):

        self.block_shape = block_shape
        self.labels = labels
        self.affinities = affinities
        self.labels_mask = labels_mask
        self.affinities_mask = affinities_mask

    def setup(self):

        assert self.labels in self.spec, (
            "Upstream does not provide %s needed by "
            "AddAffinities" % self.labels)

        spec = self.spec[self.labels].copy()
        spec.dtype = np.float32

        self.provides(self.affinities, spec)
        if self.affinities_mask:
            self.provides(self.affinities_mask, spec)
        self.enable_autoskip()

    def prepare(self, request):

        if self.labels_mask:
            assert (
                request[self.labels].roi ==
                request[self.labels_mask].roi), (
                "requested GT label roi %s and GT label mask roi %s are not "
                "the same." % (
                    request[self.labels].roi,
                    request[self.labels_mask].roi))

        labels_roi = request[self.labels].roi
        logger.debug("downstream %s request: " % self.labels + str(labels_roi))

    def process(self, batch, request):

        labels = batch.arrays[self.labels].data.astype('uint64', copy=False)
        if 0 in labels:
            labels += 1

        # mask the labels if we request a mask
        if self.affinities_mask and self.affinities_mask in request:
            if self.labels_mask:
                logger.debug("computing ground-truth affinities mask from "
                             "labels mask")
                mask = batch.arrays[self.labels_mask].data.astype('bool', copy=False)
                labels[mask] = 0
        else:
            if self.labels_mask is not None:
                logger.warning("GT labels does have a mask, but affinities "
                               "mask is not requested.")

        logger.debug("computing ground-truth affinities from labels")
        affinities, affinities_mask = compute_multiscale_affinities(labels,
                                                                    self.block_shape,
                                                                    True, 0)

        spec = self.spec[self.affinities].copy()
        # TODO do we need a roi ?
        # if yes, the labels_roi doesn't make sense, because we have downsampled ...
        # spec.roi = labels_roi
        batch.arrays[self.affinities] = Array(affinities, spec)

        if self.affinities_mask and self.affinities_mask in request:
            batch.arrays[self.affinities_mask] = Array(affinities_mask.astype('float32'), spec)

        batch.block_shape = self.block_shape
