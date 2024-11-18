void qmp_blockdev_change_medium(bool has_device, const char *device,

                                bool has_id, const char *id,

                                const char *filename,

                                bool has_format, const char *format,

                                bool has_read_only,

                                BlockdevChangeReadOnlyMode read_only,

                                Error **errp)

{

    BlockBackend *blk;

    BlockDriverState *medium_bs = NULL;

    int bdrv_flags;

    int rc;

    QDict *options = NULL;

    Error *err = NULL;



    blk = qmp_get_blk(has_device ? device : NULL,

                      has_id ? id : NULL,

                      errp);

    if (!blk) {

        goto fail;

    }



    if (blk_bs(blk)) {

        blk_update_root_state(blk);

    }



    bdrv_flags = blk_get_open_flags_from_root_state(blk);

    bdrv_flags &= ~(BDRV_O_TEMPORARY | BDRV_O_SNAPSHOT | BDRV_O_NO_BACKING |

        BDRV_O_PROTOCOL);



    if (!has_read_only) {

        read_only = BLOCKDEV_CHANGE_READ_ONLY_MODE_RETAIN;

    }



    switch (read_only) {

    case BLOCKDEV_CHANGE_READ_ONLY_MODE_RETAIN:

        break;



    case BLOCKDEV_CHANGE_READ_ONLY_MODE_READ_ONLY:

        bdrv_flags &= ~BDRV_O_RDWR;

        break;



    case BLOCKDEV_CHANGE_READ_ONLY_MODE_READ_WRITE:

        bdrv_flags |= BDRV_O_RDWR;

        break;



    default:

        abort();

    }



    if (has_format) {

        options = qdict_new();

        qdict_put(options, "driver", qstring_from_str(format));

    }



    medium_bs = bdrv_open(filename, NULL, options, bdrv_flags, errp);

    if (!medium_bs) {

        goto fail;

    }



    bdrv_add_key(medium_bs, NULL, &err);

    if (err) {

        error_propagate(errp, err);

        goto fail;

    }



    rc = do_open_tray(has_device ? device : NULL,

                      has_id ? id : NULL,

                      false, &err);

    if (rc && rc != -ENOSYS) {

        error_propagate(errp, err);

        goto fail;

    }

    error_free(err);

    err = NULL;



    qmp_x_blockdev_remove_medium(has_device, device, has_id, id, &err);

    if (err) {

        error_propagate(errp, err);

        goto fail;

    }



    qmp_blockdev_insert_anon_medium(blk, medium_bs, &err);

    if (err) {

        error_propagate(errp, err);

        goto fail;

    }



    blk_apply_root_state(blk, medium_bs);



    qmp_blockdev_close_tray(has_device, device, has_id, id, errp);



fail:

    /* If the medium has been inserted, the device has its own reference, so

     * ours must be relinquished; and if it has not been inserted successfully,

     * the reference must be relinquished anyway */

    bdrv_unref(medium_bs);

}
