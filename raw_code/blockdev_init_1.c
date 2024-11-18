static DriveInfo *blockdev_init(QemuOpts *all_opts,

                                BlockInterfaceType block_default_type)

{

    const char *buf;

    const char *file = NULL;

    const char *serial;

    const char *mediastr = "";

    BlockInterfaceType type;

    enum { MEDIA_DISK, MEDIA_CDROM } media;

    int bus_id, unit_id;

    int cyls, heads, secs, translation;

    int max_devs;

    int index;

    int ro = 0;

    int bdrv_flags = 0;

    int on_read_error, on_write_error;

    const char *devaddr;

    DriveInfo *dinfo;

    ThrottleConfig cfg;

    int snapshot = 0;

    bool copy_on_read;

    int ret;

    Error *error = NULL;

    QemuOpts *opts;

    QDict *bs_opts;

    const char *id;

    bool has_driver_specific_opts;

    BlockDriver *drv = NULL;



    translation = BIOS_ATA_TRANSLATION_AUTO;

    media = MEDIA_DISK;



    /* Check common options by copying from all_opts to opts, all other options

     * are stored in bs_opts. */

    id = qemu_opts_id(all_opts);

    opts = qemu_opts_create(&qemu_common_drive_opts, id, 1, &error);

    if (error_is_set(&error)) {

        qerror_report_err(error);

        error_free(error);

        return NULL;

    }



    bs_opts = qdict_new();

    qemu_opts_to_qdict(all_opts, bs_opts);

    qemu_opts_absorb_qdict(opts, bs_opts, &error);

    if (error_is_set(&error)) {

        qerror_report_err(error);

        error_free(error);

        return NULL;

    }



    if (id) {

        qdict_del(bs_opts, "id");

    }



    has_driver_specific_opts = !!qdict_size(bs_opts);



    /* extract parameters */

    bus_id  = qemu_opt_get_number(opts, "bus", 0);

    unit_id = qemu_opt_get_number(opts, "unit", -1);

    index   = qemu_opt_get_number(opts, "index", -1);



    cyls  = qemu_opt_get_number(opts, "cyls", 0);

    heads = qemu_opt_get_number(opts, "heads", 0);

    secs  = qemu_opt_get_number(opts, "secs", 0);



    snapshot = qemu_opt_get_bool(opts, "snapshot", 0);

    ro = qemu_opt_get_bool(opts, "read-only", 0);

    copy_on_read = qemu_opt_get_bool(opts, "copy-on-read", false);



    file = qemu_opt_get(opts, "file");

    serial = qemu_opt_get(opts, "serial");



    if ((buf = qemu_opt_get(opts, "if")) != NULL) {

        for (type = 0; type < IF_COUNT && strcmp(buf, if_name[type]); type++)

            ;

        if (type == IF_COUNT) {

            error_report("unsupported bus type '%s'", buf);

            return NULL;

	}

    } else {

        type = block_default_type;

    }



    max_devs = if_max_devs[type];



    if (cyls || heads || secs) {

        if (cyls < 1) {

            error_report("invalid physical cyls number");

	    return NULL;

	}

        if (heads < 1) {

            error_report("invalid physical heads number");

	    return NULL;

	}

        if (secs < 1) {

            error_report("invalid physical secs number");

	    return NULL;

	}

    }



    if ((buf = qemu_opt_get(opts, "trans")) != NULL) {

        if (!cyls) {

            error_report("'%s' trans must be used with cyls, heads and secs",

                         buf);

            return NULL;

        }

        if (!strcmp(buf, "none"))

            translation = BIOS_ATA_TRANSLATION_NONE;

        else if (!strcmp(buf, "lba"))

            translation = BIOS_ATA_TRANSLATION_LBA;

        else if (!strcmp(buf, "auto"))

            translation = BIOS_ATA_TRANSLATION_AUTO;

	else {

            error_report("'%s' invalid translation type", buf);

	    return NULL;

	}

    }



    if ((buf = qemu_opt_get(opts, "media")) != NULL) {

        if (!strcmp(buf, "disk")) {

	    media = MEDIA_DISK;

	} else if (!strcmp(buf, "cdrom")) {

            if (cyls || secs || heads) {

                error_report("CHS can't be set with media=%s", buf);

	        return NULL;

            }

	    media = MEDIA_CDROM;

	} else {

	    error_report("'%s' invalid media", buf);

	    return NULL;

	}

    }



    if ((buf = qemu_opt_get(opts, "discard")) != NULL) {

        if (bdrv_parse_discard_flags(buf, &bdrv_flags) != 0) {

            error_report("invalid discard option");

            return NULL;

        }

    }



    if (qemu_opt_get_bool(opts, "cache.writeback", true)) {

        bdrv_flags |= BDRV_O_CACHE_WB;

    }

    if (qemu_opt_get_bool(opts, "cache.direct", false)) {

        bdrv_flags |= BDRV_O_NOCACHE;

    }

    if (qemu_opt_get_bool(opts, "cache.no-flush", false)) {

        bdrv_flags |= BDRV_O_NO_FLUSH;

    }



#ifdef CONFIG_LINUX_AIO

    if ((buf = qemu_opt_get(opts, "aio")) != NULL) {

        if (!strcmp(buf, "native")) {

            bdrv_flags |= BDRV_O_NATIVE_AIO;

        } else if (!strcmp(buf, "threads")) {

            /* this is the default */

        } else {

           error_report("invalid aio option");

           return NULL;

        }

    }

#endif



    if ((buf = qemu_opt_get(opts, "format")) != NULL) {

        if (is_help_option(buf)) {

            error_printf("Supported formats:");

            bdrv_iterate_format(bdrv_format_print, NULL);

            error_printf("\n");

            return NULL;

        }



        drv = bdrv_find_whitelisted_format(buf, ro);

        if (!drv) {

            if (!ro && bdrv_find_whitelisted_format(buf, !ro)) {

                error_report("'%s' can be only used as read-only device.", buf);

            } else {

                error_report("'%s' invalid format", buf);

            }

            return NULL;

        }

    }



    /* disk I/O throttling */

    memset(&cfg, 0, sizeof(cfg));

    cfg.buckets[THROTTLE_BPS_TOTAL].avg =

        qemu_opt_get_number(opts, "throttling.bps-total", 0);

    cfg.buckets[THROTTLE_BPS_READ].avg  =

        qemu_opt_get_number(opts, "throttling.bps-read", 0);

    cfg.buckets[THROTTLE_BPS_WRITE].avg =

        qemu_opt_get_number(opts, "throttling.bps-write", 0);

    cfg.buckets[THROTTLE_OPS_TOTAL].avg =

        qemu_opt_get_number(opts, "throttling.iops-total", 0);

    cfg.buckets[THROTTLE_OPS_READ].avg =

        qemu_opt_get_number(opts, "throttling.iops-read", 0);

    cfg.buckets[THROTTLE_OPS_WRITE].avg =

        qemu_opt_get_number(opts, "throttling.iops-write", 0);



    cfg.buckets[THROTTLE_BPS_TOTAL].max =

        qemu_opt_get_number(opts, "throttling.bps-total-max", 0);

    cfg.buckets[THROTTLE_BPS_READ].max  =

        qemu_opt_get_number(opts, "throttling.bps-read-max", 0);

    cfg.buckets[THROTTLE_BPS_WRITE].max =

        qemu_opt_get_number(opts, "throttling.bps-write-max", 0);

    cfg.buckets[THROTTLE_OPS_TOTAL].max =

        qemu_opt_get_number(opts, "throttling.iops-total-max", 0);

    cfg.buckets[THROTTLE_OPS_READ].max =

        qemu_opt_get_number(opts, "throttling.iops-read-max", 0);

    cfg.buckets[THROTTLE_OPS_WRITE].max =

        qemu_opt_get_number(opts, "throttling.iops-write-max", 0);



    cfg.op_size = qemu_opt_get_number(opts, "throttling.iops-size", 0);



    if (!check_throttle_config(&cfg, &error)) {

        error_report("%s", error_get_pretty(error));

        error_free(error);

        return NULL;

    }



    if (qemu_opt_get(opts, "boot") != NULL) {

        fprintf(stderr, "qemu-kvm: boot=on|off is deprecated and will be "

                "ignored. Future versions will reject this parameter. Please "

                "update your scripts.\n");

    }



    on_write_error = BLOCKDEV_ON_ERROR_ENOSPC;

    if ((buf = qemu_opt_get(opts, "werror")) != NULL) {

        if (type != IF_IDE && type != IF_SCSI && type != IF_VIRTIO && type != IF_NONE) {

            error_report("werror is not supported by this bus type");

            return NULL;

        }



        on_write_error = parse_block_error_action(buf, 0);

        if (on_write_error < 0) {

            return NULL;

        }

    }



    on_read_error = BLOCKDEV_ON_ERROR_REPORT;

    if ((buf = qemu_opt_get(opts, "rerror")) != NULL) {

        if (type != IF_IDE && type != IF_VIRTIO && type != IF_SCSI && type != IF_NONE) {

            error_report("rerror is not supported by this bus type");

            return NULL;

        }



        on_read_error = parse_block_error_action(buf, 1);

        if (on_read_error < 0) {

            return NULL;

        }

    }



    if ((devaddr = qemu_opt_get(opts, "addr")) != NULL) {

        if (type != IF_VIRTIO) {

            error_report("addr is not supported by this bus type");

            return NULL;

        }

    }



    /* compute bus and unit according index */



    if (index != -1) {

        if (bus_id != 0 || unit_id != -1) {

            error_report("index cannot be used with bus and unit");

            return NULL;

        }

        bus_id = drive_index_to_bus_id(type, index);

        unit_id = drive_index_to_unit_id(type, index);

    }



    /* if user doesn't specify a unit_id,

     * try to find the first free

     */



    if (unit_id == -1) {

       unit_id = 0;

       while (drive_get(type, bus_id, unit_id) != NULL) {

           unit_id++;

           if (max_devs && unit_id >= max_devs) {

               unit_id -= max_devs;

               bus_id++;

           }

       }

    }



    /* check unit id */



    if (max_devs && unit_id >= max_devs) {

        error_report("unit %d too big (max is %d)",

                     unit_id, max_devs - 1);

        return NULL;

    }



    /*

     * catch multiple definitions

     */



    if (drive_get(type, bus_id, unit_id) != NULL) {

        error_report("drive with bus=%d, unit=%d (index=%d) exists",

                     bus_id, unit_id, index);

        return NULL;

    }



    /* init */



    dinfo = g_malloc0(sizeof(*dinfo));

    if ((buf = qemu_opts_id(opts)) != NULL) {

        dinfo->id = g_strdup(buf);

    } else {

        /* no id supplied -> create one */

        dinfo->id = g_malloc0(32);

        if (type == IF_IDE || type == IF_SCSI)

            mediastr = (media == MEDIA_CDROM) ? "-cd" : "-hd";

        if (max_devs)

            snprintf(dinfo->id, 32, "%s%i%s%i",

                     if_name[type], bus_id, mediastr, unit_id);

        else

            snprintf(dinfo->id, 32, "%s%s%i",

                     if_name[type], mediastr, unit_id);

    }

    dinfo->bdrv = bdrv_new(dinfo->id);

    dinfo->bdrv->open_flags = snapshot ? BDRV_O_SNAPSHOT : 0;

    dinfo->bdrv->read_only = ro;

    dinfo->devaddr = devaddr;

    dinfo->type = type;

    dinfo->bus = bus_id;

    dinfo->unit = unit_id;

    dinfo->cyls = cyls;

    dinfo->heads = heads;

    dinfo->secs = secs;

    dinfo->trans = translation;

    dinfo->opts = all_opts;

    dinfo->refcount = 1;

    if (serial != NULL) {

        dinfo->serial = g_strdup(serial);

    }

    QTAILQ_INSERT_TAIL(&drives, dinfo, next);



    bdrv_set_on_error(dinfo->bdrv, on_read_error, on_write_error);



    /* disk I/O throttling */

    if (throttle_enabled(&cfg)) {

        bdrv_io_limits_enable(dinfo->bdrv);

        bdrv_set_io_limits(dinfo->bdrv, &cfg);

    }



    switch(type) {

    case IF_IDE:

    case IF_SCSI:

    case IF_XEN:

    case IF_NONE:

        dinfo->media_cd = media == MEDIA_CDROM;

        break;

    case IF_SD:

    case IF_FLOPPY:

    case IF_PFLASH:

    case IF_MTD:

        break;

    case IF_VIRTIO:

    {

        /* add virtio block device */

        QemuOpts *devopts;

        devopts = qemu_opts_create_nofail(qemu_find_opts("device"));

        if (arch_type == QEMU_ARCH_S390X) {

            qemu_opt_set(devopts, "driver", "virtio-blk-s390");

        } else {

            qemu_opt_set(devopts, "driver", "virtio-blk-pci");

        }

        qemu_opt_set(devopts, "drive", dinfo->id);

        if (devaddr)

            qemu_opt_set(devopts, "addr", devaddr);

        break;

    }

    default:

        abort();

    }

    if (!file || !*file) {

        if (has_driver_specific_opts) {

            file = NULL;

        } else {

            return dinfo;

        }

    }

    if (snapshot) {

        /* always use cache=unsafe with snapshot */

        bdrv_flags &= ~BDRV_O_CACHE_MASK;

        bdrv_flags |= (BDRV_O_SNAPSHOT|BDRV_O_CACHE_WB|BDRV_O_NO_FLUSH);

    }



    if (copy_on_read) {

        bdrv_flags |= BDRV_O_COPY_ON_READ;

    }



    if (runstate_check(RUN_STATE_INMIGRATE)) {

        bdrv_flags |= BDRV_O_INCOMING;

    }



    if (media == MEDIA_CDROM) {

        /* CDROM is fine for any interface, don't check.  */

        ro = 1;

    } else if (ro == 1) {

        if (type != IF_SCSI && type != IF_VIRTIO && type != IF_FLOPPY &&

            type != IF_NONE && type != IF_PFLASH) {

            error_report("read-only not supported by this bus type");

            goto err;

        }

    }



    bdrv_flags |= ro ? 0 : BDRV_O_RDWR;



    if (ro && copy_on_read) {

        error_report("warning: disabling copy_on_read on read-only drive");

    }



    QINCREF(bs_opts);

    ret = bdrv_open(dinfo->bdrv, file, bs_opts, bdrv_flags, drv, &error);



    if (ret < 0) {

        error_report("could not open disk image %s: %s",

                     file ?: dinfo->id, error_get_pretty(error));

        goto err;

    }



    if (bdrv_key_required(dinfo->bdrv))

        autostart = 0;



    QDECREF(bs_opts);

    qemu_opts_del(opts);



    return dinfo;



err:

    qemu_opts_del(opts);

    QDECREF(bs_opts);

    bdrv_unref(dinfo->bdrv);

    g_free(dinfo->id);

    QTAILQ_REMOVE(&drives, dinfo, next);

    g_free(dinfo);

    return NULL;

}
