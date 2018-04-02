import pathlib

from clusterapp.features import Audio


class Library:
    def __init__(self, path):
        self.segments = {}

        for file in pathlib.Path(path).iterdir():
            audio = Audio(file)
            category = file.name.split('-')[0]
            items = self.segments.get(category, [])
            items.append(audio)
            self.segments[category] = items

        self.categories = set(self.segments.keys())

    def get_features(self, species, features):
        return [{
            'name': category,
            'data': [{
                'name': audio.name,
                'x': float(getattr(audio, features[0]).mean()),
                'y': float(getattr(audio, features[1]).mean())
            } for audio in self.segments[category]]
        } for category in species]
