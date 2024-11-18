static int v9fs_synth_remove(FsContext *ctx, const char *path)

{

    errno = EPERM;

    return -1;

}
