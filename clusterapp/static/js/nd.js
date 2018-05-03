var features_dimensions = {
    'min_freq': 1,
    'max_freq': 1,
    'peak_freq': 1,
    'peak_ampl': 1,
    'fundamental_freq': 1,
    'bandwidth': 1,
    'mfcc': 13
};

function getSelectedFeatures() {
    var features = [];
    $('#features-filter').find('option:selected').each(function (index, sp) {
        features.push(sp.value);
    });
    return features;
}

$(function () {
    $('#2d-bar').removeClass('active');
    $('#nd-bar').addClass('active');
    $('#best-features-bar').removeClass('active');

    $('#features-filter').select2({
        containerCssClass: "no-border"
    });
});
