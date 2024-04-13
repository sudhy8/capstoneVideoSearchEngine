from flask import Flask, Response
import json
import time
from flask_cors import CORS
from threading import Thread, Event

app = Flask(__name__)
CORS(app)

# Flag to track if the stream should be stopped
stop_stream = Event()

def stop_stream_after_10_seconds():
    stop_stream.wait(3)
    stop_stream.set()

@app.route('/stream')
def stream():
    if stop_stream.is_set():
        stop_stream.clear()

    def generate_data():
        while not stop_stream.is_set():
            data = {
                'time': time.time(),
                'value': 10 * (1 + 0.1 * (time.time() % 10))
            }
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(1)

    # Start the thread to stop the stream after 10 seconds
    stop_stream_thread = Thread(target=stop_stream_after_10_seconds)
    stop_stream_thread.start()

    return Response(generate_data(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
