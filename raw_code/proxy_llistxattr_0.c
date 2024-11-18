static ssize_t proxy_llistxattr(FsContext *ctx, V9fsPath *fs_path,

                                void *value, size_t size)

{

    int retval;

    retval = v9fs_request(ctx->private, T_LLISTXATTR, value, "ds", size,

                        fs_path);

    if (retval < 0) {

        errno = -retval;

    }

    return retval;

}
