from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

app = Flask(__name__)
filename = 'MY_MODEL.pkl'
model= pickle.load(open(filename, 'rb'))
cv = pickle.load(open('vectorizers.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        #input_vectorized = vectorizer.transform(message)
        data=[message]
        vect=cv.transform(data).toarray()
        my_pred=model.predict(vect)
        if(my_pred==0):
            re="FAKE NEWS"
        else:
            re="REAL NEWS"
        return render_template('result.html',prediction=re)



if __name__ == '__main__':
    app.run(debug=True)
