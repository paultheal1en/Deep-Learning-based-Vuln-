static int local_mkdir(FsContext *ctx, const char *path, mode_t mode)

{

    return mkdir(rpath(ctx, path), mode);

}
