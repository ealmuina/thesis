import argparse
import time

from flask import Flask, request, render_template, jsonify

from clusterapp.core import Library, statistics

app = Flask(__name__)


@app.route('/')
def index():
    axis = [
        ('min_freq', 'Min Frequency (Hz)'),
        ('max_freq', 'Max Frequency (Hz)'),
        ('zcr', 'Zero Crossing Rate'),
        ('peak_ampl', 'Peak Amplitude'),
        ('spectral_centroid', 'Spectral Centroid (Hz)'),
        ('bandwidth', 'Bandwidth (Hz)')
    ]
    clustering_algorithms = [
        ('none', 'None'),
        ('kmeans', 'K-Means'),
        ('spectral', 'Spectral Clustering'),
        ('gmm', 'Gaussian Mixture Model'),
        ('hdbscan', 'HDBSCAN'),
        ('affinity', 'Affinity Propagation')
    ]
    return render_template('index.html', **{
        'axis': axis,
        'clustering_algorithms': clustering_algorithms
    })


@app.route('/parameters_2d')
def parameters_2d():
    x = request.args.get('x')
    y = request.args.get('y')
    clustering_algorithm = request.args.get('clustering_algorithm')
    species = request.args.getlist('species[]')

    clustering, scores = LIBRARY.cluster(species, (x, y), clustering_algorithm)
    stats = statistics(clustering)

    return jsonify(
        segments=[{
            'name': label if label != '-1' else 'noise',
            'data': [{
                'name': item['name'],
                'x': item['x'][0],
                'y': item['x'][1]
            } for item in clustering[label]],
            'statistics': stats[label]
        } for label in clustering.keys()],
        scores=scores
    )


@app.route('/search_for_species')
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
    args = parser.parse_args()

    start = time.time()
    LIBRARY = Library(args.path)
    print('Features computed in %.3f seconds.' % (time.time() - start))

    app.run(host=args.host, port=args.port)
