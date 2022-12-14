import os, sys
from flask import *
import cv2
import tensorflow as tf
from keras.backend import set_session, get_session
from engine import PredictEngine

global sess
global graph

sess = tf.Session()
graph = tf.get_default_graph()

engine = PredictEngine()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/"

@app.route("/")
def upload():
    return render_template("index.html")

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        print(full_filename)
        image_ext = cv2.imread(full_filename)
        img_path = full_filename
        with graph.as_default():
            set_session(sess)
            text = engine.predict(img_path)

        final_text = 'Result'
        return render_template("show.html", name = final_text, img = full_filename, out_1 = text)
 

if __name__ == '__main__':  
    app.run(host="127.0.0.1", port=8080, debug=True)