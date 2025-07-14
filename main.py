from flask import Flask
from app import views
app=Flask(__name__)
app.add_url_rule(rule='/',endpoint='home',view_func=views.index)
app.add_url_rule(rule='/app/',endpoint='app',view_func=views.app)
app.add_url_rule(rule='/app/gender/',endpoint='gender',view_func=views.genderapp,methods=['GET', 'POST'])

# Add direct routes for /gender and /age to fix button errors
app.add_url_rule(rule='/gender', endpoint='gender_direct', view_func=views.genderapp, methods=['GET', 'POST'])
app.add_url_rule(rule='/age', endpoint='age_direct', view_func=lambda: views.render_template('age.html'))



if __name__ == '__main__':
    app.run(debug=True)