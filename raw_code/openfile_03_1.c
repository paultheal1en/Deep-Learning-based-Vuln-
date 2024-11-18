static int openfile(char *name, int flags, bool writethrough, bool force_share,

                    QDict *opts)

{

    Error *local_err = NULL;

    BlockDriverState *bs;



    if (qemuio_blk) {

        error_report("file open already, try 'help close'");

        QDECREF(opts);

        return 1;

    }



    if (force_share) {

        if (!opts) {

            opts = qdict_new();

        }

        if (qdict_haskey(opts, BDRV_OPT_FORCE_SHARE)

            && !qdict_get_bool(opts, BDRV_OPT_FORCE_SHARE)) {

            error_report("-U conflicts with image options");

            QDECREF(opts);

            return 1;

        }

        qdict_put_bool(opts, BDRV_OPT_FORCE_SHARE, true);

    }

    qemuio_blk = blk_new_open(name, NULL, opts, flags, &local_err);

    if (!qemuio_blk) {

        error_reportf_err(local_err, "can't open%s%s: ",

                          name ? " device " : "", name ?: "");

        return 1;

    }



    bs = blk_bs(qemuio_blk);

    if (bdrv_is_encrypted(bs) && bdrv_key_required(bs)) {

        char password[256];

        printf("Disk image '%s' is encrypted.\n", name);

        if (qemu_read_password(password, sizeof(password)) < 0) {

            error_report("No password given");

            goto error;

        }

        if (bdrv_set_key(bs, password) < 0) {

            error_report("invalid password");

            goto error;

        }

    }



    blk_set_enable_write_cache(qemuio_blk, !writethrough);



    return 0;



 error:

    blk_unref(qemuio_blk);

    qemuio_blk = NULL;

    return 1;

}
