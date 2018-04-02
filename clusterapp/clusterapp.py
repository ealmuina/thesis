from flask import Flask, request, render_template, jsonify

from clusterapp.core import load_segments

app = Flask(__name__)


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
    return render_template('index.html', **{
        'axis': axis
    })


@app.route('/parameters_2d')
def parameters_2d():
    x = request.args.get('x')
    y = request.args.get('y')

    return jsonify({
        'segments': load_segments(
            path='/home/eddy/PycharmProjects/thesis/sounds/testing/',
            features=(x, y)
        )
    })


if __name__ == '__main__':
    app.run()
