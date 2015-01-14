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
    <h3>Please enter your text here, or upload it.</h3>
What book are we looking at today?
<form action="http://www.google.com">
    <textarea cols="50" rows="5">
    </textarea>
  </body>
</html>
'''