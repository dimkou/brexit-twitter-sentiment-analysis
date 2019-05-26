var dayWidth = 200;
var stemWidth = 2;
var maxColumnHeight = 250;
var maxDonateLength = 50;
var maxSumValue = 0;
var maxDonate = 0;
var donateStartAngle = 15;
var margin = 60;
var zoom;

window.onresize = fit;

var dom = {}

function initDom() {
    dom.svg = d3.select('#graph');
    dom.container = d3.select('#container');
    dom.dataContainer = d3.select('#data');
    dom.days = d3.select('#days');
    dom.backs = d3.select('#backs');
    dom.foregs = d3.select('#foregs');
}

function calcConstants() {
    maxSumValue = d3.max(timelineData, function(d) { return d.sum; });
    maxDonate = d3.max(timelineData, function(d) {
        return d3.max(d.donates, function(d) { return d.sum; });
    });
}

function draw() {
    initDom();
    calcConstants();

    drawAxis();
    drawDays();

    zoom = d3.behavior.zoom()
        .on("zoom", zoomHandler);

    fit();

    zoom
        .scale(1)
        .translate([dom.dX, 0]);

    dom.svg
        .call(zoom) // delete this line to disable free zooming
        .call(zoom.event);
}

function fit() {
    dom.width = timelineData.length * dayWidth;
    dom.allWidth = document.getElementById('svgContainer').offsetWidth;
    dom.allHeight = document.getElementById('svgContainer').offsetHeight;
    dom.dX = (dom.allWidth - dom.width) / 2;
    dom.dY = dom.allHeight - (dom.allHeight - maxColumnHeight - maxDonateLength) / 2;

    if (dom.dX < margin) {
        dom.dX = margin;
        dom.svg.classed('move', true);
    }
    else
        dom.svg.classed('move', false);

    dom.container.attr('transform', 'translate(0, ' + dom.dY + ')');

    dom.leftAxis.attr('x2', -dom.dX);
    dom.rightAxis.attr('x1', dom.width);
    dom.rightAxis.attr('x2', dom.width + dom.dX);

    var topBgHeight = dom.dY > 400 ? 400 : dom.dY;
    var bottomBgHeight = dom.allHeight - dom.dY > 100 ? 100 : dom.allHeight - dom.dY;
    d3.selectAll('.topBackground')
        .attr('y', -topBgHeight)
        .attr('height', topBgHeight);

    d3.selectAll('.bottomBackground')
        .attr('height', bottomBgHeight);

    d3.selectAll('.leftBackground')
        .attr('x', -dom.dX)
        .attr('width', dom.dX);

    d3.selectAll('.rightBackground')
        .attr('x', dom.width)
        .attr('width', dom.dX);

    dom.foregrounds
        .attr('y', -dom.dY)
        .attr('height', dom.allHeight);
}

function getHeight(d) {
    return d.sum * maxColumnHeight / maxSumValue;
}

function drawAxis() {
    dom.leftAxis = d3.select('#timeAxis').append('line')
        .attr('x1', 0)
        .attr('y1', 0)
        .attr('x2', 100)
        .attr('y2', 0);
    dom.rightAxis = d3.select('#timeAxis').append('line')
        .attr('x1', 0)
        .attr('y1', 0)
        .attr('x2', 100)
        .attr('y2', 0);

    dom.days = dom.days.selectAll('.day')
            .data(timelineData)
        .enter().append('g')
            .classed('date', true);


    dom.days.filter(function(d, i) { return i + 1}).append('text')
        .classed('day', true)
        .attr('x', function(d, i) {
            return (i + 0.5) * dayWidth;
        })
        .attr('y', 20)
        .text(function(d) {
            return ['22 February 2016', '23 June 2016', 
                    '19 June 2017', '14 November 2018',
                    '25 November 2018', '15 January 2019'][d.month];
        });
}

function drawDays() {
    dom.dayColumns = dom.dataContainer.selectAll('.dayColumn')
            .data(timelineData)
        .enter().append('g')
            .classed('dayColumn', true)
            .attr('transform', function(d, i) {
                return 'translate(' + i * dayWidth + ',0)';
            });

    dom.backgrounds = dom.backs.selectAll('.background')
            .data(timelineData)
        .enter().append('g')
            .classed('background', true)
            .attr('transform', function(d, i) {
                return 'translate(' + i * dayWidth + ',0)';
            });

    dom.topBackgrounds = dom.backgrounds.append('rect')
        .classed('topBackground', true)
        .attr('y', 0)
        .attr('width', dayWidth)
        .attr('height', 100);

    dom.bottomBackgrounds = dom.backgrounds.append('rect')
        .classed('bottomBackground', true)
        .attr('y', 0)
        .attr('width', dayWidth)
        .attr('height', 100);

    dom.backs.append('rect')
        .classed('bottomBackground leftBackground', true)
        .attr('y', 0);
    dom.backs.append('rect')
        .classed('topBackground leftBackground', true)
        .attr('y', 0);
    dom.backs.append('rect')
        .classed('bottomBackground rightBackground', true)
        .attr('y', 0);
    dom.backs.append('rect')
        .classed('topBackground rightBackground', true)
        .attr('y', 0);


    dom.dayColumns.append('path')
        .attr('d', function(d, i) {
            if (d.sum > 0) {
                var res = 'M0,0 ';
                res += 'Q' + 0.5 * (dayWidth - stemWidth) + ',0 ';
                res += 0.5 * (dayWidth - stemWidth) + ',-10 ';
                res += 'L' + 0.5 * (dayWidth - stemWidth) + ',-' + getHeight(d) + ' ';
                res += 'L' + 0.5 * (dayWidth + stemWidth) + ',-' + getHeight(d) + ' ';
                res += 'L' + 0.5 * (dayWidth + stemWidth) + ',-10 ';
                res += 'Q' + 0.5 * (dayWidth + stemWidth) + ',0 ';
                res += dayWidth + ',0';
                return res;
            }
            else
                return 'M0,0 L' + dayWidth + ',0';
        });

    drawDonates(dom.dayColumns.append('g')
        .classed('flower', true)
        .attr('transform', function(d, i) {0
            return 'translate(' + 0.5 * dayWidth + ',-' + getHeight(d) + ')';
        }));

    drawInfo(dom.dayColumns.filter(function(g) { return g.sum > 0; }).append('g')
        .classed('info', true)
        .attr('transform', function(d, i) {0
            return 'translate(' + 0.5 * dayWidth + ',-' + (getHeight(d) + 90) + ')';
        }));

    dom.foregrounds = dom.foregs.selectAll('.foreground')
            .data(timelineData)
        .enter().append('rect')
        .classed('foreground', true)
        .attr('x', function(d, i) { return (i + 0.5) * dayWidth; })
        .attr('width', 100)
        .on('mouseover', selectDay)
        .on('mouseout', deselectDay)
        .on('click', modal);
}

function drawInfo(p) {
    p.append('text')
        .classed('label sum', true)
        .text('Total number of tweets:');
    p.append('text')
        .classed('sum', true)
        .attr('x', 5)
        .text(function(d) { return d.sum });

    p.append('text')
        .classed('label', true)
        .text('Positive tweets:')
        .attr('y', 16);
    p.append('text')
        .attr('x', 5)
        .text(function(d) { return d.pos; })
        .attr('y', 16);

    p.append('text')
        .classed('label', true)
        .text('Negative tweets:')
        .attr('y', 28);
    p.append('text')
        .attr('x', 5)
        .text(function(d) { return d.neg; })
        .attr('y', 28);
}

function drawDonates(p) {
    p.selectAll('.donate')
            .data(function(d) {return new Array(100);})
        .enter().append('line')
            .classed('donate', true)
            .attr('stroke', function(d) {
                return '#4BC6D6';
            })
            .attr('x1', 0)
            .attr('y1', 0)
            .attr('x2', 0)
            .attr('y2', function(d) {
                return 8;
            })
            .attr('opacity', function(d) {
                return 0.6;
            })
            .attr('transform', function(d, i) {
                var count = 95;
                return 'rotate(' + (donateStartAngle + 360 / count * i) + ')';
            });
}

function getRandomInt(max) {
  return Math.floor(Math.random() * Math.floor(max));
}

function modal(dToSelect) {
    var idx = getRandomInt(dToSelect.tweets.length);
    console.log(dToSelect.tweets.length);
    console.log(idx);
    var tweet = Object.keys(dToSelect.tweets[idx])[0];
    var label = dToSelect.tweets[idx][tweet] ? "positive" : "negative";
    window.alert("Tweet: " + tweet + "\nLabel: " + label)
} 

function selectDay(dToSelect) {
    d3.selectAll('.selected')
        .classed('selected', false);

    dom.dayColumns.filter(function(d) { return d == dToSelect; })
        .classed('selected', true);
    dom.backgrounds.filter(function(d) { return d == dToSelect; })
        .classed('selected', true);
    dom.days.filter(function(d) { return d == dToSelect; })
        .classed('selected', true);
}

function deselectDay(dToSelect) {
    dom.dayColumns.filter(function(d) { return d == dToSelect; })
        .classed('selected', false);
    dom.backgrounds.filter(function(d) { return d == dToSelect; })
        .classed('selected', false);
    dom.days.filter(function(d) { return d == dToSelect; })
        .classed('selected', false);
}


var oldX = 0;
function zoomHandler() {
    var newX = d3.event.translate[0];
    if (oldX == newX)
        return;

    if (newX > dom.dX || dom.width + 2 * dom.dX < dom.allWidth) {
        newX = dom.dX;
    }
    else if (dom.allWidth - dom.width - dom.dX - newX > 0) {
        newX = dom.allWidth - dom.width - dom.dX;
    }
    oldX = newX;

    dom.svg.select("#everything").attr("transform", "translate(" + newX + ",0)");
    zoom.translate([oldX, 0]);
}



Number.prototype.formatMoney = function(c, d, t){
    var n = this, 
    c = isNaN(c = Math.abs(c)) ? 0 : c, 
    d = d == undefined ? "," : d, 
    t = t == undefined ? " " : t, 
    s = n < 0 ? "-" : "", 
    i = parseInt(n = Math.abs(+n || 0).toFixed(c)) + "", 
    j = (j = i.length) > 3 ? j % 3 : 0;
    return s + (j ? i.substr(0, j) + t : "") + i.substr(j).replace(/(\d{3})(?=\d)/g, "$1" + t) + (c ? d + Math.abs(n - i).toFixed(c).slice(2) : "");
 };
