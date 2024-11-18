static int local_open(FsContext *ctx, V9fsPath *fs_path,

                      int flags, V9fsFidOpenState *fs)

{

    char buffer[PATH_MAX];

    char *path = fs_path->data;



    fs->fd = open(rpath(ctx, path, buffer), flags | O_NOFOLLOW);

    return fs->fd;

}
