static int local_chmod(FsContext *fs_ctx, const char *path, FsCred *credp)

{

    if (fs_ctx->fs_sm == SM_MAPPED) {

        return local_set_xattr(rpath(fs_ctx, path), credp);

    } else if (fs_ctx->fs_sm == SM_PASSTHROUGH) {

        return chmod(rpath(fs_ctx, path), credp->fc_mode);

    }

    return -1;

}
