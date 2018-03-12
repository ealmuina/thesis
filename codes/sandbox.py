import essentia.standard as es
import numpy as np
import pylab as pl
from matplotlib import collections as mc

NS = {
    'mpeg7': 'urn:mpeg:mpeg7:schema:2001',
    'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
}


def plot_basic_descriptors(root):
    awf = root.find(".//*[@xsi:type='AudioWaveformType']", NS)
    awf_min = map(float, awf.find(".//mpeg7:Min", NS).text.split())
    awf_max = map(float, awf.find(".//mpeg7:Max", NS).text.split())
    # samples = int(awf.find(".//mpeg7:SeriesOfScalar", ns).get('totalNumOfSamples'))
    awf = [[(i, curr_min), (i, curr_max)] for i, (curr_min, curr_max) in enumerate(zip(awf_min, awf_max))]

    ap = root.find(".//*[@xsi:type='AudioPowerType']", NS)
    ap = map(float, ap.find('.//mpeg7:Raw', NS).text.split())
    ap = np.array(list(ap))

    fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(12, 4))
    fig.subplots_adjust(left=0.07, right=0.97)

    lc = mc.LineCollection(awf, linewidths=2)
    ax1.add_collection(lc)
    ax1.autoscale()
    ax1.set_title('Audio Waveform (AWF)')

    ax2.plot(ap)
    ax2.set_title('Audio Power (AP)')

    fig.show()


def plot_basic_spectral_descriptors(root):
    pass


def main():
    # p = Popen(['java', '-jar', 'mpeg7enc/MPEG7AudioEnc-0.4-rc3.jar', '../sounds/sheep.wav', 'mpeg7enc/mpeg7config.xml'],
    #           stdout=PIPE, stderr=STDOUT)
    # result = []
    # for line in p.stdout:
    #     line = line.decode()
    #     if line[0] != '<':
    #         continue
    #     result.append(line)
    # result = ''.join(result)
    # root = ET.fromstring(result)
    #
    # plot_basic_descriptors(root)

    extractor = es.MusicExtractor()
    audio = es.MonoLoader(filename='../sounds/sheep.wav')()

    features, features_frames = extractor('../sounds/sheep.wav')
    # print(*sorted(features.descriptorNames()), sep='\n')
    print(features_frames['lowlevel.barkbands_flatness_db'].shape)


if __name__ == '__main__':
    main()
