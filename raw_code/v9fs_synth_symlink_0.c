static int v9fs_synth_symlink(FsContext *fs_ctx, const char *oldpath,

                              V9fsPath *newpath, const char *buf, FsCred *credp)

{

    errno = EPERM;

    return -1;

}
