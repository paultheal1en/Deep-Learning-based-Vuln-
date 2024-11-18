static int raw_open_common(BlockDriverState *bs, QDict *options,

                           int bdrv_flags, int open_flags, Error **errp)

{

    BDRVRawState *s = bs->opaque;

    QemuOpts *opts;

    Error *local_err = NULL;

    const char *filename;

    int fd, ret;



    opts = qemu_opts_create_nofail(&raw_runtime_opts);

    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (error_is_set(&local_err)) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto fail;

    }



    filename = qemu_opt_get(opts, "filename");



    ret = raw_normalize_devicepath(&filename);

    if (ret != 0) {

        error_setg_errno(errp, -ret, "Could not normalize device path");

        goto fail;

    }



    s->open_flags = open_flags;

    raw_parse_flags(bdrv_flags, &s->open_flags);



    s->fd = -1;

    fd = qemu_open(filename, s->open_flags, 0644);

    if (fd < 0) {

        ret = -errno;

        if (ret == -EROFS) {

            ret = -EACCES;

        }

        goto fail;

    }

    s->fd = fd;



#ifdef CONFIG_LINUX_AIO

    if (raw_set_aio(&s->aio_ctx, &s->use_aio, bdrv_flags)) {

        qemu_close(fd);

        ret = -errno;

        error_setg_errno(errp, -ret, "Could not set AIO state");

        goto fail;

    }

#endif



    s->has_discard = 1;

#ifdef CONFIG_XFS

    if (platform_test_xfs_fd(s->fd)) {

        s->is_xfs = 1;

    }

#endif



    ret = 0;

fail:

    qemu_opts_del(opts);

    return ret;

}
