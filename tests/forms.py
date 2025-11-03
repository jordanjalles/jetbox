# forms.py
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, TextAreaField, SubmitField, HiddenField
from wtforms.validators import DataRequired, Email, EqualTo, Length

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=25)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class PostForm(FlaskForm):
    title = StringField('Title', validators=[DataRequired(), Length(max=200)])
    body = TextAreaField('Body', validators=[DataRequired()])
    submit = SubmitField('Submit')

class CommentForm(FlaskForm):
    body = TextAreaField('Comment', validators=[DataRequired()])
    post_id = HiddenField('Post ID', validators=[DataRequired()])
    submit = SubmitField('Add Comment')
