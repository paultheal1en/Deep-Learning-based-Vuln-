static int local_post_create_passthrough(FsContext *fs_ctx, const char *path,

        FsCred *credp)

{

    if (chmod(rpath(fs_ctx, path), credp->fc_mode & 07777) < 0) {

        return -1;

    }

    if (chown(rpath(fs_ctx, path), credp->fc_uid, credp->fc_gid) < 0) {

        return -1;

    }

    return 0;

}
