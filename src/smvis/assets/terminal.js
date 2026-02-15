/* Auto-scroll terminal output to bottom when content changes */
(function() {
    var observer = new MutationObserver(function() {
        var el = document.getElementById('terminal-output');
        if (el) {
            el.scrollTop = el.scrollHeight;
        }
    });

    function attach() {
        var el = document.getElementById('terminal-output');
        if (el) {
            observer.observe(el, { childList: true, characterData: true, subtree: true });
        } else {
            setTimeout(attach, 500);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attach);
    } else {
        attach();
    }
})();
