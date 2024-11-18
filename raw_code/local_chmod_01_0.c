static int local_chmod(FsContext *fs_ctx, V9fsPath *fs_path, FsCred *credp)

{

    char buffer[PATH_MAX];

    char *path = fs_path->data;



    if (fs_ctx->fs_sm == SM_MAPPED) {

        return local_set_xattr(rpath(fs_ctx, path, buffer), credp);

    } else if ((fs_ctx->fs_sm == SM_PASSTHROUGH) ||

               (fs_ctx->fs_sm == SM_NONE)) {

        return chmod(rpath(fs_ctx, path, buffer), credp->fc_mode);

    }

    return -1;

}
