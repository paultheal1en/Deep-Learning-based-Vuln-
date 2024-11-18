static ssize_t local_readlink(FsContext *fs_ctx, const char *path,

        char *buf, size_t bufsz)

{

    ssize_t tsize = -1;

    if (fs_ctx->fs_sm == SM_MAPPED) {

        int fd;

        fd = open(rpath(fs_ctx, path), O_RDONLY);

        if (fd == -1) {

            return -1;

        }

        do {

            tsize = read(fd, (void *)buf, bufsz);

        } while (tsize == -1 && errno == EINTR);

        close(fd);

        return tsize;

    } else if (fs_ctx->fs_sm == SM_PASSTHROUGH) {

        tsize = readlink(rpath(fs_ctx, path), buf, bufsz);

    }

    return tsize;

}
