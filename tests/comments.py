# comments.py
from flask import Blueprint, request, redirect, url_for, flash, abort
from flask_login import login_required, current_user
from app import db
from models import Comment, Post
from forms import CommentForm

bp = Blueprint('comments', __name__, url_prefix='/comments')

@bp.route('/create', methods=['POST'])
@login_required
def create():
    form = CommentForm()
    if form.validate_on_submit():
        post = Post.query.get(form.post_id.data)
        if not post:
            flash('Post not found.')
            return redirect(url_for('posts.index'))
        comment = Comment(body=form.body.data, author=current_user, post=post)
        db.session.add(comment)
        db.session.commit()
        flash('Comment added.')
        return redirect(url_for('posts.detail', post_id=post.id))
    flash('Invalid comment.')
    return redirect(url_for('posts.index'))

@bp.route('/<int:comment_id>/delete', methods=['POST'])
@login_required
def delete(comment_id):
    comment = Comment.query.get_or_404(comment_id)
    if comment.author != current_user:
        abort(403)
    post_id = comment.post_id
    db.session.delete(comment)
    db.session.commit()
    flash('Comment deleted.')
    return redirect(url_for('posts.detail', post_id=post_id))
