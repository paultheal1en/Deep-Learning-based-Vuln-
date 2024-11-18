static ssize_t proxy_lgetxattr(FsContext *ctx, V9fsPath *fs_path,

                               const char *name, void *value, size_t size)

{

    int retval;

    V9fsString xname;



    v9fs_string_init(&xname);

    v9fs_string_sprintf(&xname, "%s", name);

    retval = v9fs_request(ctx->private, T_LGETXATTR, value, "dss", size,

                          fs_path, &xname);

    v9fs_string_free(&xname);

    if (retval < 0) {

        errno = -retval;

    }

    return retval;

}
