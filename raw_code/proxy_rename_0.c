static int proxy_rename(FsContext *ctx, const char *oldpath,

                        const char *newpath)

{

    int retval;

    V9fsString oldname, newname;



    v9fs_string_init(&oldname);

    v9fs_string_init(&newname);



    v9fs_string_sprintf(&oldname, "%s", oldpath);

    v9fs_string_sprintf(&newname, "%s", newpath);

    retval = v9fs_request(ctx->private, T_RENAME, NULL, "ss",

                          &oldname, &newname);

    v9fs_string_free(&oldname);

    v9fs_string_free(&newname);

    if (retval < 0) {

        errno = -retval;

    }

    return retval;

}
