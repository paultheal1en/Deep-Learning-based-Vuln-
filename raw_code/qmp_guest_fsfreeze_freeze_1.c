int64_t qmp_guest_fsfreeze_freeze(Error **err)

{

    int ret = 0, i = 0;

    FsMountList mounts;

    struct FsMount *mount;

    Error *local_err = NULL;

    int fd;



    slog("guest-fsfreeze called");



    execute_fsfreeze_hook(FSFREEZE_HOOK_FREEZE, &local_err);

    if (error_is_set(&local_err)) {

        error_propagate(err, local_err);

        return -1;

    }



    QTAILQ_INIT(&mounts);

    build_fs_mount_list(&mounts, &local_err);

    if (error_is_set(&local_err)) {

        error_propagate(err, local_err);

        return -1;

    }



    /* cannot risk guest agent blocking itself on a write in this state */

    ga_set_frozen(ga_state);



    QTAILQ_FOREACH(mount, &mounts, next) {

        fd = qemu_open(mount->dirname, O_RDONLY);

        if (fd == -1) {

            error_setg_errno(err, errno, "failed to open %s", mount->dirname);

            goto error;

        }



        /* we try to cull filesytems we know won't work in advance, but other

         * filesytems may not implement fsfreeze for less obvious reasons.

         * these will report EOPNOTSUPP. we simply ignore these when tallying

         * the number of frozen filesystems.

         *

         * any other error means a failure to freeze a filesystem we

         * expect to be freezable, so return an error in those cases

         * and return system to thawed state.

         */

        ret = ioctl(fd, FIFREEZE);

        if (ret == -1) {

            if (errno != EOPNOTSUPP) {

                error_setg_errno(err, errno, "failed to freeze %s",

                                 mount->dirname);

                close(fd);

                goto error;

            }

        } else {

            i++;

        }

        close(fd);

    }



    free_fs_mount_list(&mounts);

    return i;



error:

    free_fs_mount_list(&mounts);

    qmp_guest_fsfreeze_thaw(NULL);

    return 0;

}
