
//$(document).ready(function() {
//    $('.dataTables-dict').dataTable({
//        responsive: true,
////        "dom": 'T<"clear">lfrtip',
//        "bFilter": false,
//        "bPaginate": false,
//        "bInfo":false
//    });
//    $('.dataTables-full').dataTable({
//        responsive: true,
//        "dom": 'T<"clear">lfrtip',
//        "tableTools": {
//            "sSwfPath": "/static/cjs/plugins/dataTables/swf/copy_csv_xls_pdf.swf"
//        }
//    });
//    var ageOptions = {
//        series: {
//            bars: {
//                show: true,
//                barWidth: 0.6,
//                align: 'center',
//                fill: true,
//                fillColor: {
//                    colors: [{
//                        opacity: 0.8
//                    }, {
//                        opacity: 0.8
//                    }]
//                }
//            }
//        },
//        xaxis: {
//            tickDecimals: 0
//        },
//        colors: ["#1ab394"],
//        grid: {
//            color: "#999999",
//            hoverable: true,
//            clickable: true,
//            tickColor: "#D4D4D4",
//            borderWidth:0
//        },
//        legend: {
//            show: false
//        },
//        tooltip: true,
//        tooltipOpts: {
//            content: "Age in years: %x, Visits: %y"
//        }
//    };
//    var ageData = {
//        label: "bar",
//        data: [
//            {% for k in payload.entry.stats.ageh.h %}
//                    [{{k.k}},{{k.v}}],
//            {% endfor %}
//        ]
//    };
//    $.plot($("#age-chart"), [ageData], ageOptions);
//
//
//
//
//        var losOptions = {
//        series: {
//            bars: {
//                show: true,
//                barWidth: 0.6,
//                align: 'center',
//                fill: true,
//                fillColor: {
//                    colors: [{
//                        opacity: 0.8
//                    }, {
//                        opacity: 0.8
//                    }]
//                }
//            }
//        },
//        xaxis: {
//            tickDecimals: 0
//        },
//        colors: ["#2a6496"],
//        grid: {
//            color: "#999999",
//            hoverable: true,
//            clickable: true,
//            tickColor: "#D4D4D4",
//            borderWidth:0
//        },
//        legend: {
//            show: false
//        },
//        tooltip: true,
//        tooltipOpts: {
//            content: "Length of Stay in days: %x, Visits: %y"
//        }
//    };
//    var losData = {
//        label: "bar",
//        data: [
//            {% for k in payload.entry.stats.losh.h %}
//                    [{{k.k}},{{k.v}}],
//            {% endfor %}
//        ]
//    };
//    $.plot($("#los-chart"), [losData], losOptions);
//    var barData = {
//        label: "bar",
//        data: [
//            {% for k in payload.entry.stats.ageh.h %}
//                    [{{k.k}},{{k.v}}],
//            {% endfor %}
//        ]
//    };
//
//
//
//        var yearOptions = {
//        series: {
//            bars: {
//                show: true,
//                barWidth: 0.6,
//                align: 'center',
//                fill: true,
//                fillColor: {
//                    colors: [{
//                        opacity: 0.8
//                    }, {
//                        opacity: 0.8
//                    }]
//                }
//            }
//        },
//        xaxis: {
//            tickDecimals: 0
//        },
//        colors: ["#464f88"],
//        grid: {
//            color: "#999999",
//            hoverable: true,
//            clickable: true,
//            tickColor: "#D4D4D4",
//            borderWidth:0
//        },
//        legend: {
//            show: false
//        },
//        tooltip: true,
//        tooltipOpts: {
//            content: "During %x, Visits: %y"
//        }
//    };
//    var yearData = {
//        label: "bar",
//        data: [
//            {% for k in payload.entry.stats.yearh %}
//                    [{{k.k}},{{k.v}}],
//            {% endfor %}
//        ]
//    };
//    $.plot($("#year-chart"), [yearData], yearOptions);
//    var deltaOptions = {
//    series: {
//        bars: {
//            show: true,
//            barWidth: 0.6,
//            align: 'center',
//            fill: true,
//            fillColor: {
//                colors: [{
//                    opacity: 0.8
//                }, {
//                    opacity: 0.8
//                }]
//            }
//        }
//    },
//    xaxis: {
//        tickDecimals: 0
//    },
//    colors: ["#1ab394"],
//    grid: {
//        color: "#999999",
//        hoverable: true,
//        clickable: true,
//        tickColor: "#D4D4D4",
//        borderWidth:0
//    },
//    legend: {
//        show: false
//    },
//    tooltip: true,
//    tooltipOpts: {
//        content: " &Delta;T = %x number of visits %y "
//    }
//    };
//    var deltaData = {
//        label: "bar",
//    };
//    $.plot($("#delta-chart"), [deltaData], deltaOptions);
//});
