__author__ = 'JulietteSeive'


from flask import render_template
from app import app

@app.route('/')
@app.route('/index')
def index():
    user = {'nickname': 'Miguel'}  # fake user
    return '''
<html>
  <head>
    <subtitle> Let's Disentangle Narratives!</subtitle>
  </head>
  <body>
    <h3>Please enter your text here or choose to upload it.</h3>
What book are we looking at today?
<form action="http://www.google.com">
    <textarea cols="50" rows="5">
    </textarea>
<br>
Number of Topics: <input title="Number of desired topics" id="num_topics" name="topics" type="text" size ="3" />

<input type="submit"/>
</form>
</br>
<br>
<form name="myWebForm" action="mailto:julietteseive@gmail.com" method="post">
<input type="file" name="uploadField" />
<br>
Number of Topics: <input title="Number of desired topics" id="num_topics" name="topics" type="text" size ="3" />
<input type="submit" value="Submit" />
</form>
</br>

  </body>
</html>
'''