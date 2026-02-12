"""Entry point: python -m nuxmv_viz"""
import sys
import webbrowser
from nuxmv_viz.app import create_app


def main():
    port = 8050
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    app = create_app()
    print(f"Starting nuXmv Model Visualizer at http://localhost:{port}")
    webbrowser.open(f"http://localhost:{port}")
    app.run(debug=True, port=port, use_reloader=False)


if __name__ == "__main__":
    main()
