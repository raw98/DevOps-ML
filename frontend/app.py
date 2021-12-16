from flask import Flask
from flask import render_template
app=Flask(__name__, template_folder='interface')


@app.route('/service')
def upload_files():
   return render_template("main.html")

		
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081) 