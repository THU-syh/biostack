/**
 * Created by aub3 on 3/17/15.
 */



MGChart = function(element,chart_data){
    this.element = element;
    this._element =  $(element);
//    console.log(chart_data[_this.data('chart-data')]);
    plot = $.plot(this._element,
        [ { data: chart_data[this._element.data('chart-data')], label: this._element.data('y-label') +" vs "+ this._element.data('x-label') } ],
        {
            series: {
                lines: { show: true, lineWidth: 0},
                bars: { show: true, barWidth: 0.5, align: "center",lineWidth:0},
                points: {show: false}
            },
            xaxis :{},
            yaxis :{ min:0},
            crosshair: { mode: "x" },
            grid: { hoverable: true, tooltip: true, clickable: true, borderWidth: 0},
            legend: {show: false},
            colors:["#0022FF"]
        });
    };


DeltaChart = function (element,delta_data,plots,displaystring_function,high,xmax) {
    xmax = typeof xmax !== 'undefined' ? xmax : 50;
    this.element = element;
    this.previousPoint = null;
    this.plots = plots;
    this._element = $(element);
    this._element.css({'margin-top':'0px'});
    this.plot = $.plot(this._element,
            [ { data: delta_data, label: "count of visits"} ],
            {
                series: {
                    lines: { show: true, lineWidth: 0},
                    bars: { show: true, barWidth: 0.5, align: "center",lineWidth:0},
                    points: {show: false}
                },
                xaxis :{ min:-0.5, max: xmax },
                yaxis :{ min:0, labelWidth: 50},
                crosshair: { mode: "x" },
                grid: { hoverable: true, tooltip: true, clickable: true, borderWidth: 0 },
                legend: {show: false},
                colors:["#0022FF"]
            });
    plots[element] = this.plot;
    var i, j, dataset = this.plot.getData();
    this.dataset = dataset;
    this.series = dataset[0];
    this.bind();
    this.plot.element = element;
    this.top = this._element.offset().top;
    this.left = this._element.offset().left;
    this.displaystring = displaystring_function;
    this.plot.displaystring = this.displaystring;
    this.plot.show_value = this.show_value;
    this.plot._element = this._element;
    this.high = high;
    this.tooltip =  $('<div id="'+this.element.slice(1)+'_tooltip"><h4 style="text-align:left;margin-top:0px;" id="'+this.element.slice(1)+'_tooltext">&NonBreakingSpace;</h4></div>').css( {
        width:this._element.width()-20,
        height: 30,
        padding: "0 0 0 60px"
     });
    $(this.element).before(this.tooltip);
    this.tooltext = $(this.element+"_tooltext");
};


DeltaChart.prototype.bind = function()
    {
    var that = this;
    $(this.element).bind("plothover", function (event, pos, item){
    if (item)
        {
            that.high.highlight(item);
        }
        else
        {
            var min = null;
            var max = null;
            for (j = 0; j < that.series.data.length; ++j)
            {
                if ( that.series.data[j+1] &&  ((that.series.data[j][0]+that.series.data[j+1][0])/2  >= pos.x))
                {
                    break;
                }
            }
            if (that.previousPoint != j && that.series.data[j])
            {
                item = { dataIndex:j, datapoint:that.series.data[j]};
//                that.highlight_value(item);
                that.high.highlight(item);
            }
        }
    });
};

DeltaChart.prototype.clear = function(){
        this.tooltext.text('\xa0');
};

DeltaChart.prototype.show_value = function(item)
{
    this.clear();
    this.tooltext.text(this.displaystring(item)).fadeIn(0);
};

DeltaChart.prototype.highlight_value = function(i)
{
    if (this.previousPoint != i.dataIndex)
    {
        this.previousPoint = i.dataIndex;
        this.show_value(i);
    }
};

function PlotHighlighter() {
    this.test = 1;
}

PlotHighlighter.prototype.initialize = function(charts) {
    this.charts = charts;
    this.x_map = {};
    this.previousPoint = null;
    var that = this;
    _.mapObject(charts, function (chart,element) {
        plot_data = chart.plot.getData()[0].data;
        for (var oindex in plot_data)
        {
            if(plot_data.hasOwnProperty(oindex)){
                x = plot_data[oindex][0];
                if (!(x in that.x_map))
                {
                    that.x_map[x] = {};
                }
               that.x_map[x][element] = [chart,{"datapoint":plot_data[oindex]}];
            }
        }
    });
//    console.log(this.x_map);
};

PlotHighlighter.prototype.highlight = function(i){
    if (this.previousPoint != i.dataIndex)
    {
        _.mapObject(this.charts, function (chart,element) {
            chart.clear()
        });
        this.previousPoint = i.dataIndex;
        x = i.datapoint[0];
        for(element in this.x_map[x]){
            this.x_map[x][element][0].show_value(this.x_map[x][element][1]);
        }
    }
};


