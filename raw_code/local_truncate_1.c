static int local_truncate(FsContext *ctx, V9fsPath *fs_path, off_t size)

{

    char *buffer;

    int ret;

    char *path = fs_path->data;



    buffer = rpath(ctx, path);

    ret = truncate(buffer, size);

    g_free(buffer);

    return ret;

}
