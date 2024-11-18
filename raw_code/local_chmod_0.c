static int local_chmod(FsContext *ctx, const char *path, mode_t mode)

{

    return chmod(rpath(ctx, path), mode);

}
