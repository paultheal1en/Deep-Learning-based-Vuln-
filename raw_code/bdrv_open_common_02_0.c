static int bdrv_open_common(BlockDriverState *bs, BdrvChild *file,

                            QDict *options, Error **errp)

{

    int ret, open_flags;

    const char *filename;

    const char *driver_name = NULL;

    const char *node_name = NULL;

    QemuOpts *opts;

    BlockDriver *drv;

    Error *local_err = NULL;



    assert(bs->file == NULL);

    assert(options != NULL && bs->options != options);



    opts = qemu_opts_create(&bdrv_runtime_opts, NULL, 0, &error_abort);

    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto fail_opts;

    }



    driver_name = qemu_opt_get(opts, "driver");

    drv = bdrv_find_format(driver_name);

    assert(drv != NULL);



    if (file != NULL) {

        filename = file->bs->filename;

    } else {

        filename = qdict_get_try_str(options, "filename");

    }



    if (drv->bdrv_needs_filename && !filename) {

        error_setg(errp, "The '%s' block driver requires a file name",

                   drv->format_name);

        ret = -EINVAL;

        goto fail_opts;

    }



    trace_bdrv_open_common(bs, filename ?: "", bs->open_flags,

                           drv->format_name);



    node_name = qemu_opt_get(opts, "node-name");

    bdrv_assign_node_name(bs, node_name, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto fail_opts;

    }



    bs->read_only = !(bs->open_flags & BDRV_O_RDWR);



    if (use_bdrv_whitelist && !bdrv_is_whitelisted(drv, bs->read_only)) {

        error_setg(errp,

                   !bs->read_only && bdrv_is_whitelisted(drv, true)

                        ? "Driver '%s' can only be used for read-only devices"

                        : "Driver '%s' is not whitelisted",

                   drv->format_name);

        ret = -ENOTSUP;

        goto fail_opts;

    }



    assert(bs->copy_on_read == 0); /* bdrv_new() and bdrv_close() make it so */

    if (bs->open_flags & BDRV_O_COPY_ON_READ) {

        if (!bs->read_only) {

            bdrv_enable_copy_on_read(bs);

        } else {

            error_setg(errp, "Can't use copy-on-read on read-only device");

            ret = -EINVAL;

            goto fail_opts;

        }

    }



    if (filename != NULL) {

        pstrcpy(bs->filename, sizeof(bs->filename), filename);

    } else {

        bs->filename[0] = '\0';

    }

    pstrcpy(bs->exact_filename, sizeof(bs->exact_filename), bs->filename);



    bs->drv = drv;

    bs->opaque = g_malloc0(drv->instance_size);



    /* Apply cache mode options */

    update_flags_from_options(&bs->open_flags, opts);



    /* Open the image, either directly or using a protocol */

    open_flags = bdrv_open_flags(bs, bs->open_flags);

    if (drv->bdrv_file_open) {

        assert(file == NULL);

        assert(!drv->bdrv_needs_filename || filename != NULL);

        ret = drv->bdrv_file_open(bs, options, open_flags, &local_err);

    } else {

        if (file == NULL) {

            error_setg(errp, "Can't use '%s' as a block driver for the "

                       "protocol level", drv->format_name);

            ret = -EINVAL;

            goto free_and_fail;

        }

        bs->file = file;

        ret = drv->bdrv_open(bs, options, open_flags, &local_err);

    }



    if (ret < 0) {

        if (local_err) {

            error_propagate(errp, local_err);

        } else if (bs->filename[0]) {

            error_setg_errno(errp, -ret, "Could not open '%s'", bs->filename);

        } else {

            error_setg_errno(errp, -ret, "Could not open image");

        }

        goto free_and_fail;

    }



    ret = refresh_total_sectors(bs, bs->total_sectors);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not refresh total sector count");

        goto free_and_fail;

    }



    bdrv_refresh_limits(bs, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto free_and_fail;

    }



    assert(bdrv_opt_mem_align(bs) != 0);

    assert(bdrv_min_mem_align(bs) != 0);

    assert(is_power_of_2(bs->request_alignment) || bdrv_is_sg(bs));



    qemu_opts_del(opts);

    return 0;



free_and_fail:

    bs->file = NULL;

    g_free(bs->opaque);

    bs->opaque = NULL;

    bs->drv = NULL;

fail_opts:

    qemu_opts_del(opts);

    return ret;

}
