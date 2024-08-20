import os
import json

from flask import (
    Flask, jsonify, request
)

import helper


if __name__ == '__main__':
    app = Flask(__name__)
    hlp = helper.Helper()
    @app.route('/ping')
    def ping():
        if not hlp.model_is_ready:
            return jsonify(status="nor ready")
        return jsonify(status="ready")
    
    @app.route('/query', methods=['POST'])
    def query():
        if not hlp.model_is_ready or not hlp.index_is_ready:
            return json.dumps({"status": "FAISS is not initialized!"})
        suggestions, lang_check = hlp.query_handler(request)

        return jsonify(suggestions=suggestions, lang_check=lang_check)
    
    @app.route('/update_index', methods=['POST'])
    def update_index():
        index_size = hlp.index_handler(request)

        return jsonify(index_size=index_size)
    
    hlp.prepare_model()
        
