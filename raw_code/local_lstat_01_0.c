static int local_lstat(FsContext *ctx, const char *path, struct stat *stbuf)

{

    return lstat(rpath(ctx, path), stbuf);

}
