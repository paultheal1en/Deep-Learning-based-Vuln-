static int v9fs_do_chmod(V9fsState *s, V9fsString *path, mode_t mode)

{

    return s->ops->chmod(&s->ctx, path->data, mode);

}
