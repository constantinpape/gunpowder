import os

from gunpowder import *
from gunpowder.caffe import *

def predict(raw_path, out_path, net_folder):
    assert os.path.exists(raw_path)
    assert os.path.exists(net_folder)

    iteration = 10000
    prototxt = os.path.join(net_folder, 'net.prototxt')
    weights  = os.path.join('net_iter_%d.caffemodel'%iteration)

    input_size = Coordinate((84,268,268))
    output_size = Coordinate((56,56,56))

    pipeline = (
            Hdf5Source(
                    raw_path,
                    raw_dataset='volumes/raw') +
            Normalize() +
            Pad() +
            IntensityScaleShift(2, -1) +
            ZeroOutConstSections() +
            Predict(prototxt, weights, use_gpu=0) +
            Snapshot(
                    every=1,
                    output_dir=os.path.join('chunks', '%d'%iteration),
                    output_filename='chunk.hdf'
            ) +
            PrintProfilingStats() +
            Chunk(
                    BatchSpec(
                            input_size,
                            output_size
                    )
            ) +
            Snapshot(
                    every=1,
                    output_dir=os.path.join('processed', '%d'%iteration),
                    output_filename=out_path
            )
    )

    # request a "batch" of the size of the whole dataset
    with build(pipeline) as p:
        shape = p.get_spec().roi.get_shape()
        p.request_batch(
                BatchSpec(
                        shape,
                        shape - (input_size-output_size)
                )
        )


def predict_sample(sample):
    raw_path = '/groups/saalfeld/home/papec/data/mala_jan_original/raw/sample_%s+.hdf' % sample
    out_path = '/groups/saalfeld/home/papec/data/networks/malaV2_jan/predictions'
    net_folder = '/groups/saalfeld/home/papec/data/networks/malaV2_jan/sample%s+_predictions.hdf' % sample

    predict(raw_path, out_path, net_folder)


if __name__ == "__main__":
    predict('A')
