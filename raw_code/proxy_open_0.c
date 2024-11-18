static int proxy_open(FsContext *ctx, V9fsPath *fs_path,

                      int flags, V9fsFidOpenState *fs)

{

    fs->fd = v9fs_request(ctx->private, T_OPEN, NULL, "sd", fs_path, flags);

    if (fs->fd < 0) {

        errno = -fs->fd;

        fs->fd = -1;

    }

    return fs->fd;

}
