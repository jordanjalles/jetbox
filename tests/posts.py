# posts.py
from flask import Blueprint, render_template, redirect, url_for, request, flash, abort
from flask_login import login_required, current_user
from app import db
from models import Post, Comment
from forms import PostForm, CommentForm

bp = Blueprint('posts', __name__, url_prefix='/posts')

@bp.route('/')
def index():
    posts = Post.query.order_by(Post.created_at.desc()).all()
    return render_template('index.html', posts=posts)

@bp.route('/<int:post_id>')
def detail(post_id):
    post = Post.query.get_or_404(post_id)
    comment_form = CommentForm()
    comment_form.post_id.data = post.id
    return render_template('post_detail.html', post=post, comment_form=comment_form)

@bp.route('/create', methods=['GET', 'POST'])
@login_required
def create():
    form = PostForm()
    if form.validate_on_submit():
        post = Post(title=form.title.data, body=form.body.data, author=current_user)
        db.session.add(post)
        db.session.commit()
        flash('Post created.')
        return redirect(url_for('posts.detail', post_id=post.id))
    return render_template('post_form.html', form=form)

@bp.route('/<int:post_id>/edit', methods=['GET', 'POST'])
@login_required
def edit(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    form = PostForm(obj=post)
    if form.validate_on_submit():
        post.title = form.title.data
        post.body = form.body.data
        db.session.commit()
        flash('Post updated.')
        return redirect(url_for('posts.detail', post_id=post.id))
    return render_template('post_form.html', form=form, post=post)

@bp.route('/<int:post_id>/delete', methods=['POST'])
@login_required
def delete(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    db.session.delete(post)
    db.session.commit()
    flash('Post deleted.')
    return redirect(url_for('posts.index'))
