static int local_fstat(FsContext *ctx, int fd, struct stat *stbuf)

{

    return fstat(fd, stbuf);

}
