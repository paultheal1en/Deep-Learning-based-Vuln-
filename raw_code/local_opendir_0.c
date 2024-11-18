static int local_opendir(FsContext *ctx,

                         V9fsPath *fs_path, V9fsFidOpenState *fs)

{

    char buffer[PATH_MAX];

    char *path = fs_path->data;



    fs->dir = opendir(rpath(ctx, path, buffer));

    if (!fs->dir) {

        return -1;

    }

    return 0;

}
