static int v9fs_do_mknod(V9fsState *s, V9fsString *path, mode_t mode, dev_t dev)

{

    return s->ops->mknod(&s->ctx, path->data, mode, dev);

}
