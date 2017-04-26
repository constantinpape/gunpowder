from batch_filter import BatchFilter
from batch_spec import BatchSpec
from batch import Batch
from coordinate import Coordinate
import numpy as np

import logging
logger = logging.getLogger(__name__)

class Chunk(BatchFilter):
    '''Assemble a large batch by requesting smaller chunks upstream.
    '''

    def __init__(self, chunk_spec):

        self.chunk_spec_template = chunk_spec
        self.dims = self.chunk_spec_template.input_roi.dims()

        assert chunk_spec.input_roi.get_offset() == (0,)*self.dims, "The chunk spec should not have an input offset, only input/output shape and optionally output offset (relative to input)."

    def request_batch(self, batch_spec):

        logger.info("batch with input ROI " + str(batch_spec.input_roi) + " requested")

        stride = self.chunk_spec_template.output_roi.get_shape()

        begin = batch_spec.input_roi.get_begin()
        end = batch_spec.input_roi.get_end()

        batch = None
        offset = np.array(begin)
        while (offset < end).all():

            chunk_spec = BatchSpec(
                    self.chunk_spec_template.input_roi.get_shape(),
                    self.chunk_spec_template.output_roi.get_shape(),
                    offset,
                    self.chunk_spec_template.output_roi.get_offset() + Coordinate(offset),
            )

            logger.info("requesting chunk at " + str(chunk_spec.input_roi))

            chunk = self.get_upstream_provider().request_batch(chunk_spec)

            if batch is None:
                batch = self.__setup_batch(batch_spec, chunk)

            self.__fill(batch.raw, chunk.raw, batch_spec.input_roi, chunk.spec.input_roi)
            if chunk.spec.with_gt:
                self.__fill(batch.gt, chunk.gt, batch_spec.output_roi, chunk.spec.output_roi)
            if chunk.spec.with_gt_mask:
                self.__fill(batch.gt_mask, chunk.gt_mask, batch_spec.output_roi, chunk.spec.output_roi)
            if chunk.spec.with_prediction:
                self.__fill(batch.prediction, chunk.prediction, batch_spec.output_roi, chunk.spec.output_roi)

            for d in range(self.dims):
                offset[d] += stride[d]
                if offset[d] >= end[d]:
                    if d == self.dims - 1:
                        break
                    offset[d] = begin[d]
                else:
                    break

        return batch

    def __setup_batch(self, batch_spec, reference):

        batch = Batch(batch_spec)
        batch.raw = np.zeros(batch_spec.input_roi.get_shape(), reference.raw.dtype)
        if reference.spec.with_gt:
            batch.gt = np.zeros(batch.spec.output_roi.get_shape(), reference.gt.dtype)
            batch.spec.with_gt = True
        if reference.spec.with_gt_mask:
            batch.gt_mask = np.zeros(batch.spec.output_roi.get_shape(), reference.gt_mask.dtype)
            batch.spec.with_gt_mask = True
        if reference.spec.with_prediction:
            batch.prediction = np.zeros((3,) + batch.spec.output_roi.get_shape(), reference.prediction.dtype)
            batch.spec.with_prediction = True

        return batch

    def __fill(self, a, b, roi_a, roi_b):

        logger.debug("filling " + str(roi_b) + " into " + str(roi_a))

        common_roi = roi_a.intersect(roi_b)
        if common_roi is None:
            return

        common_in_a_roi = common_roi - roi_a.get_offset()
        common_in_b_roi = common_roi - roi_b.get_offset()

        a[common_in_a_roi.get_bounding_box()] = b[common_in_b_roi.get_bounding_box()]
