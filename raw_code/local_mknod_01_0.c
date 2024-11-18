static int local_mknod(FsContext *ctx, const char *path, mode_t mode, dev_t dev)

{

    return mknod(rpath(ctx, path), mode, dev);

}
