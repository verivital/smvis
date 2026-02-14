/* smvis - Resizable panel drag handle */

document.addEventListener('DOMContentLoaded', function() {
    var observer = new MutationObserver(function() {
        var handle = document.querySelector('.resize-handle');
        if (!handle || handle._resizeInit) return;
        handle._resizeInit = true;

        var leftPanel = handle.previousElementSibling;
        var dragging = false;

        handle.addEventListener('mousedown', function(e) {
            dragging = true;
            handle.classList.add('active');
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
            e.preventDefault();
        });

        document.addEventListener('mousemove', function(e) {
            if (!dragging) return;
            var containerLeft = leftPanel.parentElement.getBoundingClientRect().left;
            var newWidth = e.clientX - containerLeft;
            newWidth = Math.max(200, Math.min(newWidth, window.innerWidth * 0.6));
            leftPanel.style.width = newWidth + 'px';
            leftPanel.style.flex = 'none';
        });

        document.addEventListener('mouseup', function() {
            if (dragging) {
                dragging = false;
                handle.classList.remove('active');
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
            }
        });
    });
    observer.observe(document.body, {childList: true, subtree: true});
});
