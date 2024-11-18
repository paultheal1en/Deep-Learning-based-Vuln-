static int v9fs_synth_truncate(FsContext *ctx, V9fsPath *path, off_t offset)

{

    errno = ENOSYS;

    return -1;

}
