static int local_truncate(FsContext *ctx, V9fsPath *fs_path, off_t size)

{

    char buffer[PATH_MAX];

    char *path = fs_path->data;



    return truncate(rpath(ctx, path, buffer), size);

}
