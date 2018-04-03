from flask import Flask, request, render_template, jsonify

from clusterapp.core import Library, statistics

app = Flask(__name__)
LIBRARY = Library('/home/eddy/PycharmProjects/thesis/sounds/testing/')


@app.route('/')
def index():
    axis = [
        ('min_freq', 'Min Frequency (Hz)'),
        ('max_freq', 'Max Frequency (Hz)'),
        ('peak_freq', 'Peak Frequency (Hz)'),
        ('peak_ampl', 'Peak Amplitude'),
        ('fundamental_freq', 'Fundamental Frequency (Hz)'),
        ('bandwidth', 'Bandwidth (Hz)')
    ]
    clustering_algorithms = [
        ('none', 'None'),
        ('kmeans', 'K-Means'),
        ('gmm', 'Gaussian Mixture Model'),
        ('hdbscan', 'HDBSCAN')
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
            'data': clustering[label],
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
    app.run()
