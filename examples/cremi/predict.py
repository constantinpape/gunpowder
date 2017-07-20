import os

from gunpowder import *
from gunpowder.caffe import *

def predict(
    raw_path,
    out_folder,
    out_name,
    net_folder
):
    assert os.path.exists(raw_path), raw_path
    assert os.path.exists(net_folder), net_folder
    assert os.path.exists(out_folder), out_folder

    iteration = 300000
    prototxt = os.path.join(net_folder, 'net.prototxt')
    assert os.path.exists(prototxt)
    weights  = os.path.join(net_folder, 'net_iter_%d.caffemodel'%iteration)
    assert os.path.exists(weights), weights

    input_size = Coordinate((84, 268, 268))
    output_size = Coordinate((56, 56, 56))

    # the size of the receptive field of the network
    context = (input_size - output_size) / 2

    # a chunk request that matches the dimensions of the network, will be used
    # to chunk the whole volume into batches of this size
    chunk_request = BatchRequest()
    chunk_request.add_volume_request(VolumeTypes.RAW, input_size)
    chunk_request.add_volume_request(VolumeTypes.PRED_AFFINITIES, output_size)

    source = Hdf5Source(raw_path, datasets={VolumeTypes.RAW: 'volumes/raw'})

    # not used anymore in new prediction scripts
    #snap1 = Snapshot(
    #                every=1,
    #                output_dir=os.path.join('chunks', '%d'%iteration),
    #                output_filename='chunk.hdf'
    #        )

    # save the prediction
    snap = Snapshot(
                    every=1,
                    output_dir=out_folder,
                    output_filename=out_name
            )

    # build the pipeline
    pipeline = (

            # raw sources
            source +

            # normalize raw data
            Normalize() +

            # pad raw data
            Pad({ VolumeTypes.RAW: (100, 100, 100) }) +

            # shift to [-1, 1]
            IntensityScaleShift(2, -1) +

            ZeroOutConstSections() +

            # do the actual prediction
            Predict(prototxt, weights, use_gpu=0) +

            PrintProfilingStats() +

            Chunk(chunk_request) +

            # save
            snap
    )

    with build(pipeline) as p:

        # get the ROI of the whole RAW region from the source
        raw_roi = source.get_spec().volumes[VolumeTypes.RAW]

        # small roi for tests
        #raw_roi.set_offset((50,500,500))
        #raw_roi.set_shape(((100,324,324)))

        print "Inference with roi:"
        print raw_roi

        # request affinity predictions for the whole RAW ROI
        whole_request = BatchRequest({
                VolumeTypes.RAW: raw_roi,
                VolumeTypes.PRED_AFFINITIES: raw_roi.grow(-context, -context)
            })

        print("Requesting " + str(whole_request) + " in chunks of " + str(chunk_request))

        p.request_batch(whole_request)


def predict_sample(sample):
    raw_path = '/groups/saalfeld/home/papec/Work/data/mala_jan_original/raw/sample_%s+.hdf' % sample
    out_folder = '/groups/saalfeld/home/papec/Work/data/networks/malaV2_cremi/predictions'
    out_name = 'sample_%s+_prediction.hdf' % sample
    net_folder = '/groups/saalfeld/home/papec/Work/data/networks/malaV2_cremi'

    predict(raw_path, out_folder, out_name, net_folder)


if __name__ == "__main__":
    predict_sample('A')
