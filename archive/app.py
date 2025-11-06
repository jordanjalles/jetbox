"""
Flask application that serves a simple fractal rendering web page.

The application exposes two endpoints:

* ``/`` – serves the HTML page with a canvas and controls.
* ``/api/fractal`` – accepts query parameters describing the fractal type and rendering
  options and returns a PNG image.

The fractal generation logic lives in :mod:`fractal` and is intentionally
separated from the web layer to keep the codebase modular and testable.
"""

from flask import Flask, render_template, request, send_file, abort
import io
import logging

from fractal import generate_mandelbrot

app = Flask(__name__)

# Configure basic logging for debugging and performance monitoring
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s")

@app.route("/")
def index():
    """Render the main page.

    The template contains a canvas element where the fractal image will be
    displayed and a set of controls that allow the user to adjust the view.
    """
    return render_template("index.html")

@app.route("/api/fractal")
def api_fractal():
    """Return a PNG image of the requested fractal.

    Query parameters:
        * ``width`` – image width in pixels (default 800)
        * ``height`` – image height in pixels (default 600)
        * ``x`` – horizontal offset of the view centre (default 0.0)
        * ``y`` – vertical offset of the view centre (default 0.0)
        * ``zoom`` – zoom factor (default 1.0)
        * ``max_iter`` – maximum iterations for escape time algorithm
          (default 200)

    The endpoint validates the parameters and returns a 400 error if any
    required value is missing or invalid.
    """
    try:
        width = int(request.args.get("width", 800))
        height = int(request.args.get("height", 600))
        x = float(request.args.get("x", 0.0))
        y = float(request.args.get("y", 0.0))
        zoom = float(request.args.get("zoom", 1.0))
        max_iter = int(request.args.get("max_iter", 200))
    except ValueError as exc:
        logging.error("Invalid query parameters: %s", exc)
        abort(400, description="Invalid query parameters")

    # Generate the fractal image as a PNG in memory
    image_bytes = generate_mandelbrot(width, height, x, y, zoom, max_iter)

    return send_file(
        io.BytesIO(image_bytes),
        mimetype="image/png",
        as_attachment=False,
        download_name="fractal.png",
    )

if __name__ == "__main__":
    # Run the development server. In production a WSGI server should be used.
    app.run(debug=True, host="0.0.0.0", port=5000)
