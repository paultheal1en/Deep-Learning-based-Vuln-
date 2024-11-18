static int proxy_readdir_r(FsContext *ctx, V9fsFidOpenState *fs,

                           struct dirent *entry,

                           struct dirent **result)

{

    return readdir_r(fs->dir, entry, result);

}
