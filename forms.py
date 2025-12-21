from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, TextAreaField, FileField
from wtforms.validators import InputRequired, Length,Email
class transform(FlaskForm):
    
    image = FileField('image file for dection')
    submit = SubmitField('Submit')
class contactform(FlaskForm):
    name = StringField('What is your name',validators=[InputRequired()])
    email = StringField('what is your email',validators=[InputRequired(),Email(message='Email required')])
    body = TextAreaField('What feedback do you want to send',validators=[InputRequired(), Length(min=4)])
    
    submit = SubmitField('Submit')

class text_to_binaryForm(FlaskForm):
    text = StringField('Enter text',validators=[InputRequired()])
    submit = SubmitField('Submit')
class binary_to_text(FlaskForm):
    binary = StringField('Enter binary',validators=[InputRequired()])
    submit = SubmitField('Submit')