from flask import Flask, request

from card_reader import FIRST_NUMBER, CardReader

app = Flask(__name__)

@app.route('/')
def index():
    # Test with file locally
    card_reader = CardReader('images/credit_card_01.png')
    result = card_reader.execute()
    print("Credit Card Type: {}".format(FIRST_NUMBER[result[0]]))
    print("Credit Card #: {}".format("".join(result)))
    return '<html><body><h1>Hello, World!</h1></body></html>'

@app.route('/upload', methods=['POST'])
def upload_file():
    # Test with file uploaded
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    card_reader = CardReader(file.filename)
    result = card_reader.execute()
    return 'File uploaded successfully'
if __name__ == '__main__':
    app.run()