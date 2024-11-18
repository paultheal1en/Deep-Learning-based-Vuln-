static int proxy_link(FsContext *ctx, V9fsPath *oldpath,

                      V9fsPath *dirpath, const char *name)

{

    int retval;

    V9fsString newpath;



    v9fs_string_init(&newpath);

    v9fs_string_sprintf(&newpath, "%s/%s", dirpath->data, name);



    retval = v9fs_request(ctx->private, T_LINK, NULL, "ss", oldpath, &newpath);

    v9fs_string_free(&newpath);

    if (retval < 0) {

        errno = -retval;

        retval = -1;

    }

    return retval;

}
