static int v9fs_synth_lremovexattr(FsContext *ctx,

                                   V9fsPath *path, const char *name)

{

    errno = ENOTSUP;

    return -1;

}
