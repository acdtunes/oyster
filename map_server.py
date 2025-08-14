#!/usr/bin/env python3
"""
Simple web server for the inclusion zone drawing tool
Serves the HTML interface and handles JSON file uploads to data folder
"""

import http.server
import socketserver
import json
import os
import urllib.parse
from datetime import datetime
import webbrowser
import threading
import time
import pandas as pd

class MapHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        """Handle POST requests to save inclusion zones"""
        if self.path == '/save_zones':
            try:
                # Read the POST data
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                # Parse the JSON data
                zones_data = json.loads(post_data.decode('utf-8'))
                
                # Ensure data directory exists
                os.makedirs('data', exist_ok=True)
                
                # Always overwrite data/inclusion_zones.json
                filename = 'data/inclusion_zones.json'
                with open(filename, 'w') as f:
                    json.dump(zones_data, f, indent=2)
                
                print(f"✅ Saved {len(zones_data.get('zones', []))} zones to {filename} (overwritten)")
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    'success': True,
                    'message': f'Saved {len(zones_data.get("zones", []))} zones to {filename}',
                    'filename': filename
                }
                
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                print(f"❌ Error saving zones: {e}")
                
                # Send error response
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    'success': False,
                    'message': f'Error saving zones: {str(e)}'
                }
                
                self.wfile.write(json.dumps(response).encode('utf-8'))
        
        elif self.path == '/load_zones':
            """Load existing zones from data folder"""
            try:
                filename = 'data/inclusion_zones.json'
                if os.path.exists(filename):
                    with open(filename, 'r') as f:
                        zones_data = json.load(f)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    
                    self.wfile.write(json.dumps(zones_data).encode('utf-8'))
                    print(f"📂 Loaded existing zones from {filename}")
                else:
                    # No existing file
                    self.send_response(404)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    
                    response = {'success': False, 'message': 'No existing zones found'}
                    self.wfile.write(json.dumps(response).encode('utf-8'))
                    
            except Exception as e:
                print(f"❌ Error loading zones: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {'success': False, 'message': f'Error loading zones: {str(e)}'}
                self.wfile.write(json.dumps(response).encode('utf-8'))
        
        elif self.path == '/load_reefs':
            """Load reef data from CSV file"""
            try:
                reef_file = 'output/st_marys/reef_metrics.csv'
                if os.path.exists(reef_file):
                    df = pd.read_csv(reef_file)
                    
                    # Convert to list of dictionaries
                    reefs = df.to_dict('records')
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    
                    response = {
                        'success': True,
                        'reefs': reefs,
                        'count': len(reefs)
                    }
                    self.wfile.write(json.dumps(response).encode('utf-8'))
                    print(f"📍 Loaded {len(reefs)} reef sites from {reef_file}")
                else:
                    self.send_response(404)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    
                    response = {'success': False, 'message': f'Reef file not found: {reef_file}'}
                    self.wfile.write(json.dumps(response).encode('utf-8'))
                    
            except Exception as e:
                print(f"❌ Error loading reefs: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {'success': False, 'message': f'Error loading reefs: {str(e)}'}
                self.wfile.write(json.dumps(response).encode('utf-8'))
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        """Handle preflight requests for CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests - serve files normally"""
        if self.path == '/':
            self.path = '/draw_inclusion_zones.html'
        return super().do_GET()

def start_server(port=8000):
    """Start the web server"""
    try:
        with socketserver.TCPServer(("", port), MapHandler) as httpd:
            print("=" * 60)
            print("🗺️ INCLUSION ZONE DRAWING TOOL SERVER")
            print("=" * 60)
            print(f"🌐 Server running at: http://localhost:{port}")
            print(f"📁 Files will be saved to: {os.path.abspath('data/')}")
            print("\n✨ Opening browser...")
            print("🎯 Draw polygons around water areas and click 'Save Zones'")
            print("💾 Files will be saved directly to the data/ folder")
            print("\nPress Ctrl+C to stop the server")
            print("=" * 60)
            
            # Open browser after a short delay
            def open_browser():
                time.sleep(1)
                webbrowser.open(f'http://localhost:{port}')
            
            threading.Thread(target=open_browser, daemon=True).start()
            
            # Start server
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use. Trying port {port + 1}...")
            start_server(port + 1)
        else:
            print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    start_server()