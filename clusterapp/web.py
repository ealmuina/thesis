from flask import Flask, request, render_template, jsonify

from clusterapp.core import statistics
from clusterapp.utils import build_library

CLUSTERING_ALGORITHMS = [
    ('hdbscan', 'HDBSCAN'),
    ('gmm', 'Gaussian Mixture Model'),
    ('kmeans', 'K-Means'),
    ('spectral', 'Spectral Clustering'),
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
    ('RmsTime', 'Root Mean Square - Time (RMS-Time)', 1),
    ('PeakToPeakTime', 'Peak to Peak Time (s)', 1),
    ('StartTime', 'Start Time (s)', 1),
    ('EndTime', 'End Time (s)', 1),
    ('DistanceToMaxTime', 'Distance to Max (s)', 1),

    ('MFCC', 'MFCC', 12)
]
_SPECTRAL_FEATURES = [
    ('MaxFreq', 'Max Frequency (Hz)', 1),
    ('MinFreq', 'Min Frequency (Hz)', 1),
    ('BandwidthFreq', 'Bandwidth (Hz)', 1),
    ('PeaksAboveFreq', 'Peaks Above Frequency', 1),
    ('EntropyFreq', 'Spectral Entropy', 1),
    ('PeakFreq', 'Peak Frequency (Hz)', 1),
    ('PeakAmpFreq', 'Peak Amplitude', 1),
    ('Spectral Energy', 'Spectral Energy', 1),
    ('Flux', 'Flux', 1),
    ('Rms Freq', 'Root Mean Square - Spectrum (RMS-Spectrum)', 1),
    ('Roll Off Freq', 'Roll-Off', 1),
    ('Shannon Entropy', 'Shannon Entropy', 1),
    ('Spectral Centroid', 'Spectral Centroid', 1),
]
_LOCATIONS = ['start', 'end', 'centre', 'max', 'max_amp']

for l in _LOCATIONS:
    for name, verbose_name, dimension in _SPECTRAL_FEATURES:
        FEATURES.append((
            '%s(%s)' % (name, l),
            '%s [%s]' % (verbose_name, l),
            dimension
        ))

app = Flask(__name__)


@app.route('/best_features/')
def best_features():
    template_type = 'classified' if CLASSIFIED else 'unclassified'

    return render_template('%s_best_features.html' % template_type, **{
        'clustering_algorithms': CLUSTERING_ALGORITHMS,
        'features_number': len(FEATURES)
    })


@app.route('/best_features_nd/')
def best_features_nd():
    clustering_algorithm = request.args.get('clustering_algorithm')
    min_features = int(request.args.get('min_features'))
    max_features = int(request.args.get('max_features'))

    if CLASSIFIED:
        categories = request.args.getlist('species[]')
    else:
        categories = int(request.args.get('n_clusters'))

    if not categories:
        return jsonify({})

    clustering, scores, features = LIBRARY.best_features(
        categories=categories,
        features_set=[f for f, _, _ in FEATURES],
        algorithm=clustering_algorithm,
        min_features=min_features,
        max_features=max_features
    )
    stats = statistics(clustering)

    report = get_report(clustering, stats, scores)
    report['features'] = features
    return jsonify(report)


@app.route('/classify/', methods=['POST'])
def classify():
    features = request.form.getlist('features[]')
    file = request.files['file']
    result = CLASSIFIER.predict([file], features)[0]
    return jsonify(result)


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
    global CLASSIFIER

    features = request.args.getlist('features[]')
    if not features:
        return jsonify({})

    clustering_algorithm = request.args.get('clustering_algorithm')

    if CLASSIFIED:
        species = request.args.getlist('species[]')
        clustering, scores, CLASSIFIER = LIBRARY.cluster(species, features, clustering_algorithm)
    else:
        n_clusters = request.args.get('n_clusters')
        if not n_clusters:
            n_clusters = '0'
        clustering, scores, CLASSIFIER = LIBRARY.cluster(int(n_clusters), features, clustering_algorithm)

    stats = statistics(clustering)
    return jsonify(get_report(clustering, stats, scores))


def run(args):
    global LIBRARY, CLASSIFIED
    LIBRARY = build_library(args.path, args.classified)
    CLASSIFIED = args.classified
    app.run(host=args.host, port=args.port)


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
