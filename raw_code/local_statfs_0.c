static int local_statfs(FsContext *s, V9fsPath *fs_path, struct statfs *stbuf)

{

    char buffer[PATH_MAX];

    char *path = fs_path->data;



    return statfs(rpath(s, path, buffer), stbuf);

}
