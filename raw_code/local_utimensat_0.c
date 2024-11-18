static int local_utimensat(FsContext *s, V9fsPath *fs_path,

                           const struct timespec *buf)

{

    char buffer[PATH_MAX];

    char *path = fs_path->data;



    return qemu_utimens(rpath(s, path, buffer), buf);

}
