from .provider_test import ProviderTest
from gunpowder import *
import numpy as np

class TestSource(BatchProvider):

    def setup(self):

        for identifier in [
            VolumeTypes.GT_AFFINITIES,
            VolumeTypes.GT_MASK,
            VolumeTypes.GT_IGNORE]:

            self.provides(
                identifier,
                VolumeSpec(
                    roi=Roi((0, 0, 0), (2000, 200, 200)),
                    voxel_size=(20, 2, 2)))

    def provide(self, request):

        batch = Batch()

        roi = request[VolumeTypes.GT_AFFINITIES].roi
        shape_vx = roi.get_shape()//self.spec[VolumeTypes.GT_AFFINITIES].voxel_size

        spec = self.spec[VolumeTypes.GT_AFFINITIES].copy()
        spec.roi = roi

        batch.volumes[VolumeTypes.GT_AFFINITIES] = Volume(
                np.random.randint(
                    0, 2,
                    (3,) + shape_vx
                ),
                spec
        )
        batch.volumes[VolumeTypes.GT_MASK] = Volume(
                np.random.randint(
                    0, 2,
                    shape_vx
                ),
                spec
        )
        batch.volumes[VolumeTypes.GT_IGNORE] = Volume(
                np.random.randint(
                    0, 2,
                    shape_vx
                ),
                spec
        )

        return batch

class TestBalanceLabels(ProviderTest):

    def test_output(self):

        pipeline = TestSource() + BalanceLabels(
                {VolumeTypes.GT_AFFINITIES: VolumeTypes.LOSS_SCALE},
                {VolumeTypes.GT_AFFINITIES: [VolumeTypes.GT_MASK, VolumeTypes.GT_IGNORE]})

        with build(pipeline):

            # check correct scaling on 10 random samples
            for i in range(10):

                request = BatchRequest()
                request.add(VolumeTypes.GT_AFFINITIES, (400,30,34))
                request.add(VolumeTypes.LOSS_SCALE, (400,30,34))

                batch = pipeline.request_batch(request)

                self.assertTrue(VolumeTypes.LOSS_SCALE in batch.volumes)

                affs = batch.volumes[VolumeTypes.GT_AFFINITIES].data
                scale = batch.volumes[VolumeTypes.LOSS_SCALE].data
                mask = batch.volumes[VolumeTypes.GT_MASK].data
                ignore = batch.volumes[VolumeTypes.GT_IGNORE].data

                # combine mask and ignore
                mask *= ignore

                # make a mask on affinities
                mask = np.array([mask, mask, mask])

                self.assertTrue((scale[mask==1] > 0).all())
                self.assertTrue((scale[mask==0] == 0).all())

                num_masked_out = affs.size - mask.sum()
                num_masked_in = affs.size - num_masked_out
                num_pos = (affs*mask).sum()
                num_neg = affs.size - num_masked_out - num_pos

                frac_pos = float(num_pos)/num_masked_in if num_masked_in > 0 else 0
                frac_pos = min(0.95, max(0.05, frac_pos))
                frac_neg = 1.0 - frac_pos

                w_pos = 1.0/(2.0*frac_pos)
                w_neg = 1.0/(2.0*frac_neg)

                self.assertAlmostEqual((scale*mask*affs).sum(), w_pos*num_pos, 3)
                self.assertAlmostEqual((scale*mask*(1-affs)).sum(), w_neg*num_neg, 3)

                # check if LOSS_SCALE is omitted if not requested
                del request[VolumeTypes.LOSS_SCALE]

                batch = pipeline.request_batch(request)
                self.assertTrue(VolumeTypes.LOSS_SCALE not in batch.volumes)
