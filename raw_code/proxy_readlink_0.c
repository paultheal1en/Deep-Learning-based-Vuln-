static ssize_t proxy_readlink(FsContext *fs_ctx, V9fsPath *fs_path,

                              char *buf, size_t bufsz)

{

    int retval;

    retval = v9fs_request(fs_ctx->private, T_READLINK, buf, "sd",

                          fs_path, bufsz);

    if (retval < 0) {

        errno = -retval;

        return -1;

    }

    return strlen(buf);

}
