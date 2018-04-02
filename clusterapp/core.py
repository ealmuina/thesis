import pathlib

from clusterapp.features import Audio


def load_segments(path, features):
    segments = {}
    for file in pathlib.Path(path).iterdir():
        audio = Audio(str(file))
        category = file.name.split('-')[0]
        items = segments.get(category, [])
        items.append({
            'name': file.name,
            'x': float(getattr(audio, features[0]).mean()),
            'y': float(getattr(audio, features[1]).mean())
        })
        segments[category] = items
    return [{
        'name': category,
        'data': segments[category],
        'visible': False
    } for category in sorted(segments.keys())]
