static int nfs_file_open(BlockDriverState *bs, QDict *options, int flags,

                         Error **errp) {

    NFSClient *client = bs->opaque;

    int64_t ret;

    QemuOpts *opts;

    Error *local_err = NULL;



    opts = qemu_opts_create(&runtime_opts, NULL, 0, &error_abort);

    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (error_is_set(&local_err)) {

        error_propagate(errp, local_err);

        return -EINVAL;

    }

    ret = nfs_client_open(client, qemu_opt_get(opts, "filename"),

                          (flags & BDRV_O_RDWR) ? O_RDWR : O_RDONLY,

                          errp);

    if (ret < 0) {

        return ret;

    }

    bs->total_sectors = ret;

    return 0;

}
