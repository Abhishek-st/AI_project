from flask import Flask, request,render_template, send_file
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/download')
def download():
	file_data = 'ast.mp3'
	return send_file('ast.mp3', attachment_filename='test.mp3', as_attachment=True)


if __name__ =='__main__':
	app.run(debug=True)
