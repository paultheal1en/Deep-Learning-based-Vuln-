static void proxy_seekdir(FsContext *ctx, V9fsFidOpenState *fs, off_t off)

{

    seekdir(fs->dir, off);

}
