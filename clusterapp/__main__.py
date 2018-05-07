import argparse
import time

from flask import Flask, request, render_template, jsonify

from clusterapp.core import ClassifiedLibrary, statistics, UnclassifiedLibrary

CLUSTERING_ALGORITHMS = [
    ('kmeans', 'K-Means'),
    ('spectral', 'Spectral Clustering'),
    ('gmm', 'Gaussian Mixture Model'),
    ('hdbscan', 'HDBSCAN'),
    ('affinity', 'Affinity Propagation')
]
FEATURES = [
    ('Autocorrelation', 'Auto Correlation', 1),
    ('StdAmplitudeTime', 'Std Amplitude (Time)', 1),
    ('VarianceAmplitudeTime', 'Variance Amplitude (Time)', 1),
    ('MeanAmplitudeTime', 'Mean Amplitude (Time)', 1),
    ('TimeCentroid', 'Temporal Centroid (s)', 1),
    ('Time Energy', 'Time Energy', 1),
    ('ZeroCrossingRate', 'Zero Crossing Rate', 1),
    ('DurationTime', 'Duration (s)', 1),
    ('RmsTime', 'RMS', 1),
    ('PeakToPeakTime', 'Peak to Peak Time (s)', 1),
    ('StartTime', 'Start Time (s)', 1),
    ('EndTime', 'End Time (s)', 1),
    ('DistanceToMaxTime', 'Distance to Max (s)', 1),

    ('MaxFreq-start', 'Max Frequency [start] (Hz)', 1),
    ('MinFreq-start', 'Min Frequency [start] (Hz)', 1),
    ('BandwidthFreq-start', 'Bandwidth [start] (Hz)', 1),
    ('PeaksAboveFreq-start', 'Peaks Above Frequency [start]', 1),
    ('EntropyFreq-start', 'Spectral Entropy [start]', 1),
    ('PeakFreq-start', 'Peak Frequency [start] (Hz)', 1),
    ('PeakAmpFreq-start', 'Peak Amplitude [start]', 1),

    ('MaxFreq-end', 'Max Frequency [end] (Hz)', 1),
    ('MinFreq-end', 'Min Frequency [end] (Hz)', 1),
    ('BandwidthFreq-end', 'Bandwidth [end] (Hz)', 1),
    ('PeaksAboveFreq-end', 'Peaks Above Frequency [end]', 1),
    ('EntropyFreq-end', 'Spectral Entropy [end]', 1),
    ('PeakFreq-end', 'Peak Frequency [end] (Hz)', 1),
    ('PeakAmpFreq-end', 'Peak Amplitude [end]', 1),

    ('MaxFreq-center', 'Max Frequency [center] (Hz)', 1),
    ('MinFreq-center', 'Min Frequency [center] (Hz)', 1),
    ('BandwidthFreq-center', 'Bandwidth [center] (Hz)', 1),
    ('PeaksAboveFreq-center', 'Peaks Above Frequency [center]', 1),
    ('EntropyFreq-center', 'Spectral Entropy [center]', 1),
    ('PeakFreq-center', 'Peak Frequency [center] (Hz)', 1),
    ('PeakAmpFreq-center', 'Peak Amplitude [center]', 1),

    ('MaxFreq-max', 'Max Frequency [max] (Hz)', 1),
    ('MinFreq-max', 'Min Frequency [max] (Hz)', 1),
    ('BandwidthFreq-max', 'Bandwidth [max] (Hz)', 1),
    ('PeaksAboveFreq-max', 'Peaks Above Frequency [max]', 1),
    ('EntropyFreq-max', 'Spectral Entropy [max]', 1),
    ('PeakFreq-max', 'Peak Frequency [max] (Hz)', 1),
    ('PeakAmpFreq-max', 'Peak Amplitude [max]', 1),

    ('MaxFreq-max_amp', 'Max Frequency [max_amp] (Hz)', 1),
    ('MinFreq-max_amp', 'Min Frequency [max_amp] (Hz)', 1),
    ('BandwidthFreq-max_amp', 'Bandwidth [max_amp] (Hz)', 1),
    ('PeaksAboveFreq-max_amp', 'Peaks Above Frequency [max_amp]', 1),
    ('EntropyFreq-max_amp', 'Spectral Entropy [max_amp]', 1),
    ('PeakFreq-max_amp', 'Peak Frequency [max_amp] (Hz)', 1),
    ('PeakAmpFreq-max_amp', 'Peak Amplitude [max_amp]', 1),
]
app = Flask(__name__)


@app.route('/best_features/')
def best_features():
    return render_template('classified_best_features.html', **{
        'axis': FEATURES,
        'clustering_algorithms': CLUSTERING_ALGORITHMS
    })


@app.route('/best_features_nd/')
def best_features_nd():
    clustering_algorithm = request.args.get('clustering_algorithm')
    species = request.args.getlist('species[]')

    if not species:
        return jsonify({})

    clustering, scores, features = LIBRARY.best_features(
        categories=species,
        features_set=[f for f, _, _ in FEATURES],
        algorithm=clustering_algorithm
    )
    stats = statistics(clustering)

    report = get_report(clustering, stats, scores)
    report['features'] = features
    return jsonify(report)


def get_report(clustering, stats, scores):
    return {
        'segments': [{
            'name': label if label != '-1' else 'noise',
            'data': [{
                'name': item['name'],
                'x': item['x_2d'][0],
                'y': item['x_2d'][1]
            } for item in clustering[label]],
            'statistics': stats[label]
        } for label in clustering.keys()],
        'scores': scores,
        'feature_set': FEATURES
    }


@app.route('/')
def index():
    clustering_algorithms = list(CLUSTERING_ALGORITHMS)
    if CLASSIFIED:
        clustering_algorithms.insert(0, ('none', 'None'))

    template_type = 'classified' if CLASSIFIED else 'unclassified'

    return render_template('%s_analysis.html' % template_type, **{
        'axis': FEATURES,
        'clustering_algorithms': clustering_algorithms
    })


@app.route('/parameters_nd/')
def parameters_nd():
    features = request.args.getlist('features[]')
    if not features:
        return jsonify({})

    clustering_algorithm = request.args.get('clustering_algorithm')

    if CLASSIFIED:
        species = request.args.getlist('species[]')
        clustering, scores = LIBRARY.cluster(species, features, clustering_algorithm)
    else:
        n_clusters = request.args.get('n_clusters')
        if not n_clusters:
            n_clusters = '0'
        clustering, scores = LIBRARY.cluster(int(n_clusters), features, clustering_algorithm)

    stats = statistics(clustering)
    return jsonify(get_report(clustering, stats, scores))


@app.route('/search_for_species/')
def search_for_species():
    q = request.args.get('q')
    excluded_species = set(request.args.getlist('exclude[]'))
    l = 10
    species = [
                  species for species in LIBRARY.categories if species not in excluded_species and q in species
              ][:l]
    species.sort()
    return jsonify({
        'success': True,
        'species': [
            {'name': sp, 'id': sp} for sp in species
        ]
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--classified', action='store_true')
    args = parser.parse_args()

    CLASSIFIED = args.classified
    start = time.time()
    if CLASSIFIED:
        LIBRARY = ClassifiedLibrary(args.path)
    else:
        LIBRARY = UnclassifiedLibrary(args.path)
    print('Features computed in %.3f seconds.' % (time.time() - start))

    app.run(host=args.host, port=args.port)
