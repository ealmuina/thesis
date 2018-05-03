$(function () {
    $('#nd-bar').removeClass('active');
    $('#2d-bar').addClass('active');
    $('#best-features-bar').removeClass('active');

    $('#x-axis').select2({
        containerCssClass: "no-border"
    });
    $('#y-axis').select2({
        containerCssClass: "no-border"
    });
});