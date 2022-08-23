#!/usr/bin/python3

# standard library
import os

# nlp
import spacy

# web services
from flask import Flask, jsonify, request, flash, redirect, url_for, render_template
from werkzeug.routing import Rule
from werkzeug.utils import secure_filename

# local packages
import api.nlp as anlp

# configuring the app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'
app.url_rule_class = lambda path, **options: Rule('/nlp' + path, **options)

# spacy model
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
nlp.max_length = 2e6
nlp.add_pipe('sentencizer')

################
@app.route('/extract/<name>')
def extract(name):
    text = anlp.load(name, app.config["UPLOAD_FOLDER"], verbose=True)
    text = anlp.clean(text)
    print('tokenization...')
    insights = anlp.get_insights(text, nlp)  # use a global instance
    return jsonify(insights)


################
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and anlp.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('extract', name=filename))
    return render_template('upload.html')


if __name__=='__main__':
    app.run(debug=True, port=8055)
