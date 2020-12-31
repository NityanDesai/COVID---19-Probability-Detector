from flask import Flask, render_template, request
app = Flask(__name__)
import json
import pandas as pd

file = open('model.pkl', 'rb')
data = pd.read_csv(file, encoding= 'unicode_escape')
clf = json.load(file)
file.close()

@app.route('/', methods = ["GET", "POST"])
def hello_world():
	if request.method == "POST":
		myDict = request.form
		fever = int(myDict['fever'])
		age = int(myDict['age'])
		bodyPain = int(myDict['bodyPain'])
		runnyNose = int(myDict['runnyNose'])
		breathDiff = int(myDict['breathDiff'])
		# Code for inference
		inputFeatures = [fever, bodyPain, age, runnyNose, breathDiff]
		infProb = clf.predict_proba([inputFeatures])[0][1]
		print(infProb)
		return render_template('/show.html', inf=round(infProb*100))
	return render_template('/index.html')

@app.route("/about-me")
def admin():
    return render_template('about.html')

if __name__ == "__main__":
	app.run(debug=True)