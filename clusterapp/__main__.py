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
    ('min_freq', 'Min Frequency (Hz)', 1),
    ('max_freq', 'Max Frequency (Hz)', 1),
    ('peak_freq', 'Peak Frequency (Hz)', 1),
    ('peak_ampl', 'Peak Amplitude', 1),
    ('fundamental_freq', 'Fundamental Frequency (Hz)', 1),
    ('bandwidth', 'Bandwidth (Hz)', 1),
    ('mfcc', 'MFCC', 13)
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
