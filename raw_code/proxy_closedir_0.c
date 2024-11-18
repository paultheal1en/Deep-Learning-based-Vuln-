static int proxy_closedir(FsContext *ctx, V9fsFidOpenState *fs)

{

    return closedir(fs->dir);

}
