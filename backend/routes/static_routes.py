"""
Serve generated output images as static files.
Registered directly on the Flask app (not a blueprint).
"""

import os
from flask import send_from_directory, current_app


def register_static_routes(app):
    """Call this from create_app() to serve output images."""

    @app.route('/outputs/<filename>')
    def serve_output(filename):
        output_folder = current_app.config['OUTPUT_FOLDER']
        return send_from_directory(os.path.abspath(output_folder), filename)

    @app.route('/uploads/<path:filepath>')
    def serve_upload(filepath):
        upload_folder = current_app.config['UPLOAD_FOLDER']
        directory = os.path.abspath(os.path.join(upload_folder, os.path.dirname(filepath)))
        return send_from_directory(directory, os.path.basename(filepath))
