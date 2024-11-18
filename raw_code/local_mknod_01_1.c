static int local_mknod(FsContext *fs_ctx, const char *path, FsCred *credp)

{

    int err = -1;

    int serrno = 0;



    /* Determine the security model */

    if (fs_ctx->fs_sm == SM_MAPPED) {

        err = mknod(rpath(fs_ctx, path), SM_LOCAL_MODE_BITS|S_IFREG, 0);

        if (err == -1) {

            return err;

        }

        local_set_xattr(rpath(fs_ctx, path), credp);

        if (err == -1) {

            serrno = errno;

            goto err_end;

        }

    } else if (fs_ctx->fs_sm == SM_PASSTHROUGH) {

        err = mknod(rpath(fs_ctx, path), credp->fc_mode, credp->fc_rdev);

        if (err == -1) {

            return err;

        }

        err = local_post_create_passthrough(fs_ctx, path, credp);

        if (err == -1) {

            serrno = errno;

            goto err_end;

        }

    }

    return err;



err_end:

    remove(rpath(fs_ctx, path));

    errno = serrno;

    return err;

}
