from flask import Flask, render_template, request, redirect, url_for
import threading
from detector import ForgottenObjectDetector

app = Flask(__name__)
CONFIG_PATH = 'config.json'

detector = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global detector
    if request.method == 'POST':
        detector = ForgottenObjectDetector(CONFIG_PATH)
        detector_thread = threading.Thread(target=detector.run, daemon=True)
        detector_thread.start()
        return redirect(url_for('index'))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)