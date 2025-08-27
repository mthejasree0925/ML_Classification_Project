from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

HTML = """
<!doctype html>
<title>Classification Project</title>
<h2>Upload CSV for Classification</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file><br><br>
  <input type=submit value=Upload>
</form>
{% if columns %}
  <form method=post>
    <input type=hidden name=csv value="{{ csv }}">
    <label>Select target column:</label>
    <select name=target>
      {% for col in columns %}
        <option value="{{ col }}">{{ col }}</option>
      {% endfor %}
    </select><br><br>
    <label>Select feature columns (Ctrl+Click for multiple):</label>
    <select name=features multiple size=10>
      {% for col in columns %}
        <option value="{{ col }}">{{ col }}</option>
      {% endfor %}
    </select><br><br>
    <input type=submit value="Train Model">
  </form>
{% endif %}
{% if result %}
  <h3>Results</h3>
  <p>Train Accuracy: {{ result.train_acc }}</p>
  <p>Test Accuracy: {{ result.test_acc }}</p>
  <p>Confusion Matrix:<br>{{ result.cm }}</p>
{% endif %}
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            df = pd.read_csv(file)
            csv = df.to_csv(index=False)
            columns = df.columns.tolist()
            return render_template_string(HTML, columns=columns, csv=csv)
        elif 'csv' in request.form:
            csv = request.form['csv']
            df = pd.read_csv(pd.compat.StringIO(csv))
            target = request.form['target']
            features = request.form.getlist('features')
            X = df[features]
            y = df[target]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=1)
            clf = LogisticRegression(random_state=0)
            trained_model = clf.fit(X_train, y_train)
            y_pred = trained_model.predict(X_test)
            train_acc = accuracy_score(y_train, trained_model.predict(X_train))
            test_acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            result = {
                'train_acc': f"{train_acc:.2f}",
                'test_acc': f"{test_acc:.2f}",
                'cm': str(cm)
            }
            columns = df.columns.tolist()
            return render_template_string(HTML, columns=columns, csv=csv, result=result)
    return render_template_string(HTML)

if __name__ == '__main__':
    app.run(debug=True)