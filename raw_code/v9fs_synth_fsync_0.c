static int v9fs_synth_fsync(FsContext *ctx, int fid_type,

                            V9fsFidOpenState *fs, int datasync)

{

    errno = ENOSYS;

    return 0;

}
