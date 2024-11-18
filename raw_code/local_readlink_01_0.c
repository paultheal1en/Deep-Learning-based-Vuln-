static ssize_t local_readlink(FsContext *fs_ctx, V9fsPath *fs_path,

                              char *buf, size_t bufsz)

{

    ssize_t tsize = -1;

    char buffer[PATH_MAX];

    char *path = fs_path->data;



    if ((fs_ctx->export_flags & V9FS_SM_MAPPED) ||

        (fs_ctx->export_flags & V9FS_SM_MAPPED_FILE)) {

        int fd;

        fd = open(rpath(fs_ctx, path, buffer), O_RDONLY | O_NOFOLLOW);

        if (fd == -1) {

            return -1;

        }

        do {

            tsize = read(fd, (void *)buf, bufsz);

        } while (tsize == -1 && errno == EINTR);

        close(fd);

        return tsize;

    } else if ((fs_ctx->export_flags & V9FS_SM_PASSTHROUGH) ||

               (fs_ctx->export_flags & V9FS_SM_NONE)) {

        tsize = readlink(rpath(fs_ctx, path, buffer), buf, bufsz);

    }

    return tsize;

}
