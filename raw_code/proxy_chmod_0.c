static int proxy_chmod(FsContext *fs_ctx, V9fsPath *fs_path, FsCred *credp)

{

    int retval;

    retval = v9fs_request(fs_ctx->private, T_CHMOD, NULL, "sd",

                          fs_path, credp->fc_mode);

    if (retval < 0) {

        errno = -retval;

    }

    return retval;

}
