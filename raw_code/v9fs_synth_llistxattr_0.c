static ssize_t v9fs_synth_llistxattr(FsContext *ctx, V9fsPath *path,

                                     void *value, size_t size)

{

    errno = ENOTSUP;

    return -1;

}
