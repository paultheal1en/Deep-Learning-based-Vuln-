static int proxy_truncate(FsContext *ctx, V9fsPath *fs_path, off_t size)

{

    int retval;



    retval = v9fs_request(ctx->private, T_TRUNCATE, NULL, "sq", fs_path, size);

    if (retval < 0) {

        errno = -retval;

        return -1;

    }

    return 0;

}
