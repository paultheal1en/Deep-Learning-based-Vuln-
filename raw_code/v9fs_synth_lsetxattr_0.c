static int v9fs_synth_lsetxattr(FsContext *ctx, V9fsPath *path,

                                const char *name, void *value,

                                size_t size, int flags)

{

    errno = ENOTSUP;

    return -1;

}
