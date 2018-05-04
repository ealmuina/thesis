var table;

function getFeatureDescription(feature, feature_set) {
    for (var i = 0; i < feature_set.length; i++) {
        var f = feature_set[i];
        if (feature === f[0])
            return {
                verbose_name: f[1],
                dimension: f[2]
            }
    }
}

function getSelectedFeatures() {
    var features = [];
    $('#features-filter').find('option:selected').each(function (index, sp) {
        features.push(sp.value);
    });
    return features;
}

function refreshTable(data, features) {
    if (table) table.destroy();

    var segments = data['segments'];
    var table_head = "<th>#</th>" +
        "            <th>Main class</th>" +
        "            <th>Proportion</th>";
    var table_body = "";

    for (var i = 0; i < features.length; i++) {
        var feature = getFeatureDescription(features[i], data['feature_set']);
        for (var j = 0; j < feature.dimension; j++) {
            table_head += "<th>" + feature.verbose_name;
            if (feature.dimension > 1)
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