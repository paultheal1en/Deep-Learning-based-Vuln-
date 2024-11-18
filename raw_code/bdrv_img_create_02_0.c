void bdrv_img_create(const char *filename, const char *fmt,

                     const char *base_filename, const char *base_fmt,

                     char *options, uint64_t img_size, int flags, bool quiet,

                     Error **errp)

{

    QemuOptsList *create_opts = NULL;

    QemuOpts *opts = NULL;

    const char *backing_fmt, *backing_file;

    int64_t size;

    BlockDriver *drv, *proto_drv;

    Error *local_err = NULL;

    int ret = 0;



    /* Find driver and parse its options */

    drv = bdrv_find_format(fmt);

    if (!drv) {

        error_setg(errp, "Unknown file format '%s'", fmt);

        return;

    }



    proto_drv = bdrv_find_protocol(filename, true, errp);

    if (!proto_drv) {

        return;

    }



    if (!drv->create_opts) {

        error_setg(errp, "Format driver '%s' does not support image creation",

                   drv->format_name);

        return;

    }



    if (!proto_drv->create_opts) {

        error_setg(errp, "Protocol driver '%s' does not support image creation",

                   proto_drv->format_name);

        return;

    }



    create_opts = qemu_opts_append(create_opts, drv->create_opts);

    create_opts = qemu_opts_append(create_opts, proto_drv->create_opts);



    /* Create parameter list with default values */

    opts = qemu_opts_create(create_opts, NULL, 0, &error_abort);

    qemu_opt_set_number(opts, BLOCK_OPT_SIZE, img_size, &error_abort);



    /* Parse -o options */

    if (options) {

        qemu_opts_do_parse(opts, options, NULL, &local_err);

        if (local_err) {

            error_report_err(local_err);

            local_err = NULL;

            error_setg(errp, "Invalid options for file format '%s'", fmt);

            goto out;

        }

    }



    if (base_filename) {

        qemu_opt_set(opts, BLOCK_OPT_BACKING_FILE, base_filename, &local_err);

        if (local_err) {

            error_setg(errp, "Backing file not supported for file format '%s'",

                       fmt);

            goto out;

        }

    }



    if (base_fmt) {

        qemu_opt_set(opts, BLOCK_OPT_BACKING_FMT, base_fmt, &local_err);

        if (local_err) {

            error_setg(errp, "Backing file format not supported for file "

                             "format '%s'", fmt);

            goto out;

        }

    }



    backing_file = qemu_opt_get(opts, BLOCK_OPT_BACKING_FILE);

    if (backing_file) {

        if (!strcmp(filename, backing_file)) {

            error_setg(errp, "Error: Trying to create an image with the "

                             "same filename as the backing file");

            goto out;

        }

    }



    backing_fmt = qemu_opt_get(opts, BLOCK_OPT_BACKING_FMT);



    /* The size for the image must always be specified, unless we have a backing

     * file and we have not been forbidden from opening it. */

    size = qemu_opt_get_size(opts, BLOCK_OPT_SIZE, 0);

    if (backing_file && !(flags & BDRV_O_NO_BACKING)) {

        BlockDriverState *bs;

        char *full_backing = g_new0(char, PATH_MAX);

        int back_flags;

        QDict *backing_options = NULL;



        bdrv_get_full_backing_filename_from_filename(filename, backing_file,

                                                     full_backing, PATH_MAX,

                                                     &local_err);

        if (local_err) {

            g_free(full_backing);

            goto out;

        }



        /* backing files always opened read-only */

        back_flags = flags;

        back_flags &= ~(BDRV_O_RDWR | BDRV_O_SNAPSHOT | BDRV_O_NO_BACKING);



        if (backing_fmt) {

            backing_options = qdict_new();

            qdict_put_str(backing_options, "driver", backing_fmt);

        }



        bs = bdrv_open(full_backing, NULL, backing_options, back_flags,

                       &local_err);

        g_free(full_backing);

        if (!bs && size != -1) {

            /* Couldn't open BS, but we have a size, so it's nonfatal */

            warn_reportf_err(local_err,

                            "Could not verify backing image. "

                            "This may become an error in future versions.\n");

            local_err = NULL;

        } else if (!bs) {

            /* Couldn't open bs, do not have size */

            error_append_hint(&local_err,

                              "Could not open backing image to determine size.\n");

            goto out;

        } else {

            if (size == -1) {

                /* Opened BS, have no size */

                size = bdrv_getlength(bs);

                if (size < 0) {

                    error_setg_errno(errp, -size, "Could not get size of '%s'",

                                     backing_file);

                    bdrv_unref(bs);

                    goto out;

                }

                qemu_opt_set_number(opts, BLOCK_OPT_SIZE, size, &error_abort);

            }

            bdrv_unref(bs);

        }

    } /* (backing_file && !(flags & BDRV_O_NO_BACKING)) */



    if (size == -1) {

        error_setg(errp, "Image creation needs a size parameter");

        goto out;

    }



    if (!quiet) {

        printf("Formatting '%s', fmt=%s ", filename, fmt);

        qemu_opts_print(opts, " ");

        puts("");

    }



    ret = bdrv_create(drv, filename, opts, &local_err);



    if (ret == -EFBIG) {

        /* This is generally a better message than whatever the driver would

         * deliver (especially because of the cluster_size_hint), since that

         * is most probably not much different from "image too large". */

        const char *cluster_size_hint = "";

        if (qemu_opt_get_size(opts, BLOCK_OPT_CLUSTER_SIZE, 0)) {

            cluster_size_hint = " (try using a larger cluster size)";

        }

        error_setg(errp, "The image size is too large for file format '%s'"

                   "%s", fmt, cluster_size_hint);

        error_free(local_err);

        local_err = NULL;

    }



out:

    qemu_opts_del(opts);

    qemu_opts_free(create_opts);

    error_propagate(errp, local_err);

}
