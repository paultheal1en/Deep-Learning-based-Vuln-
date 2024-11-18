static int proxy_unlinkat(FsContext *ctx, V9fsPath *dir,

                          const char *name, int flags)

{

    int ret;

    V9fsString fullname;

    v9fs_string_init(&fullname);



    v9fs_string_sprintf(&fullname, "%s/%s", dir->data, name);

    ret = proxy_remove(ctx, fullname.data);

    v9fs_string_free(&fullname);



    return ret;

}
