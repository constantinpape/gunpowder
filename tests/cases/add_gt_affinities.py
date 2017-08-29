from .provider_test import ProviderTest
from gunpowder import *
from itertools import product
import itertools
import numpy as np
import logging

class TestSource(BatchProvider):

    def setup(self):

        self.provides(
            VolumeTypes.GT_LABELS, VolumeSpec(
                roi=Roi((-40, -40, -40), (160, 160, 160)),
                voxel_size=(20, 4, 8),
                interpolatable=False))

    def provide(self, request):

        batch = Batch()

        roi = request[VolumeTypes.GT_LABELS].roi
        shape = (roi/self.spec[VolumeTypes.GT_LABELS].voxel_size).get_shape()

        spec = self.spec[VolumeTypes.GT_LABELS].copy()
        spec.roi = roi
        batch.volumes[VolumeTypes.GT_LABELS] = Volume(
            np.random.randint(
                0, 2,
                shape
            ),
            spec
        )

        return batch

class TestAddGtAffinities(ProviderTest):

    def test_output(self):

        # skip the test if malis is not installed
        if isinstance(gunpowder.ext.malis, gunpowder.ext.NoSuchModule):
            return

        neighborhood = [
                Coordinate((-2,0,0)),
                Coordinate((0,-1,0)),
                Coordinate((0,0,1)),
                Coordinate((1,1,1))
        ]

        pipeline = (
                TestSource() +
                AddGtAffinities(neighborhood)
        )

        with build(pipeline):

            for i in range(10):

                request = BatchRequest()
                request.add(VolumeTypes.GT_LABELS, (100,16,64))
                request.add(VolumeTypes.GT_AFFINITIES, (100,16,64))

                batch = pipeline.request_batch(request)

                self.assertTrue(VolumeTypes.GT_LABELS in batch.volumes)
                self.assertTrue(VolumeTypes.GT_AFFINITIES in batch.volumes)

                labels = batch.volumes[VolumeTypes.GT_LABELS]
                affs = batch.volumes[VolumeTypes.GT_AFFINITIES]

                self.assertTrue((len(neighborhood),) + labels.data.shape == affs.data.shape)

                voxel_roi = Roi((0,0,0), labels.data.shape)
                for (z,y,x) in product(*[range(d) for d in labels.data.shape]):

                    p = Coordinate((z,y,x))

                    for n in range(len(neighborhood)):

                        pn = p + neighborhood[n]
                        if not voxel_roi.contains(pn):
                            continue

                        a = labels.data[p]
                        b = labels.data[pn]

                        if a == b and a != 0 and b != 0:
                            self.assertEqual(affs.data[(n,)+p], 1.0, "%s -> %s, %s -> %s, but is not 1"%(p, pn, a, b))
                        else:
                            self.assertEqual(affs.data[(n,)+p], 0.0, "%s -> %s, %s -> %s, but is not 0"%(p, pn, a, b))
