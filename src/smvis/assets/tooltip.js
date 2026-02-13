/* smvis - Tooltip mouse tracking and graph mouseleave handler */

// Track mouse position globally for tooltip placement
document.addEventListener('mousemove', function(e) {
    window._smvisMouseX = e.clientX;
    window._smvisMouseY = e.clientY;
});

// Hide tooltip when mouse leaves the graph container
document.addEventListener('DOMContentLoaded', function() {
    var observer = new MutationObserver(function() {
        var graph = document.getElementById('state-graph');
        if (graph && !graph._smvisListenerAdded) {
            graph.addEventListener('mouseleave', function() {
                var tooltip = document.getElementById('hover-tooltip');
                if (tooltip) {
                    tooltip.style.display = 'none';
                }
            });
            graph._smvisListenerAdded = true;
        }
    });
    observer.observe(document.body, {childList: true, subtree: true});
});
