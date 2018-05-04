var table;
var verbose_features = {
    'min_freq': 'Min Frequency (Hz)',
    'max_freq': 'Max Frequency (Hz)',
    'peak_freq': 'Peak Frequency (Hz)',
    'peak_ampl': 'Peak Amplitude',
    'fundamental_freq': 'Fundamental Frequency (Hz)',
    'bandwidth': 'Bandwidth (Hz)',
    'mfcc': 'MFCC'
};
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

function refreshTable(data, features) {
    if (table) table.destroy();

    var segments = data.segments;
    var table_head = "<th>#</th>" +
        "            <th>Main class</th>" +
        "            <th>Proportion</th>";
    var table_body = "";

    for (var i = 0; i < features.length; i++) {
        var feature = features[i];
        var dimension = features_dimensions[feature];
        for (var j = 0; j < dimension; j++) {
            table_head += "<th>" + verbose_features[feature];
            if (dimension > 1)
                table_head += " [" + j + "]";
        }
        table_head += "</th>";
    }
    $('#statistics-table-head').html(table_head);

    for (var i = 0; i < segments.length; i++) {
        var segment = segments[i];
        if (segment.name === 'noise') continue;
        table_body +=
            "<tr>" +
            "<td>" + segment.name + "</td>" +
            "<td>" + segment.statistics.label_true + "</td>" +
            "<td>" + segment.statistics.label_true_count + '/' + segment.statistics.total + "</td>";

        var mean = segment.statistics.mean;
        var std = segment.statistics.std;
        for (var j = 0; j < mean.length; j++) {
            table_body +=
                "<td>" + mean[j] + " ± " + std[j] + "</td>";
        }
        table_body += "</tr>";
    }
    $('#statistics-table-body').html(table_body);

    table = $('#statistics-table').DataTable({
        paging: false,
        searching: false
    });
}