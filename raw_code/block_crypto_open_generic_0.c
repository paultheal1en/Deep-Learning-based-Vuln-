static int block_crypto_open_generic(QCryptoBlockFormat format,

                                     QemuOptsList *opts_spec,

                                     BlockDriverState *bs,

                                     QDict *options,

                                     int flags,

                                     Error **errp)

{

    BlockCrypto *crypto = bs->opaque;

    QemuOpts *opts = NULL;

    Error *local_err = NULL;

    int ret = -EINVAL;

    QCryptoBlockOpenOptions *open_opts = NULL;

    unsigned int cflags = 0;



    bs->file = bdrv_open_child(NULL, options, "file", bs, &child_file,

                               false, errp);

    if (!bs->file) {

        return -EINVAL;

    }



    opts = qemu_opts_create(opts_spec, NULL, 0, &error_abort);

    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        goto cleanup;

    }



    open_opts = block_crypto_open_opts_init(format, opts, errp);

    if (!open_opts) {

        goto cleanup;

    }



    if (flags & BDRV_O_NO_IO) {

        cflags |= QCRYPTO_BLOCK_OPEN_NO_IO;

    }

    crypto->block = qcrypto_block_open(open_opts,

                                       block_crypto_read_func,

                                       bs,

                                       cflags,

                                       errp);



    if (!crypto->block) {

        ret = -EIO;

        goto cleanup;

    }



    bs->encrypted = true;

    bs->valid_key = true;



    ret = 0;

 cleanup:

    qapi_free_QCryptoBlockOpenOptions(open_opts);

    return ret;

}
