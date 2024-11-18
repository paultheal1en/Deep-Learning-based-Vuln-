DriveInfo *drive_init(QemuOpts *opts, int default_to_scsi, int *fatal_error)

{

    const char *buf;

    const char *file = NULL;

    char devname[128];

    const char *serial;

    const char *mediastr = "";

    BlockInterfaceType type;

    enum { MEDIA_DISK, MEDIA_CDROM } media;

    int bus_id, unit_id;

    int cyls, heads, secs, translation;

    BlockDriver *drv = NULL;

    int max_devs;

    int index;

    int ro = 0;

    int bdrv_flags = 0;

    int on_read_error, on_write_error;

    const char *devaddr;

    DriveInfo *dinfo;

    int snapshot = 0;

    int ret;



    *fatal_error = 1;



    translation = BIOS_ATA_TRANSLATION_AUTO;



    if (default_to_scsi) {

        type = IF_SCSI;

        max_devs = MAX_SCSI_DEVS;

        pstrcpy(devname, sizeof(devname), "scsi");

    } else {

        type = IF_IDE;

        max_devs = MAX_IDE_DEVS;

        pstrcpy(devname, sizeof(devname), "ide");

    }

    media = MEDIA_DISK;



    /* extract parameters */

    bus_id  = qemu_opt_get_number(opts, "bus", 0);

    unit_id = qemu_opt_get_number(opts, "unit", -1);

    index   = qemu_opt_get_number(opts, "index", -1);



    cyls  = qemu_opt_get_number(opts, "cyls", 0);

    heads = qemu_opt_get_number(opts, "heads", 0);

    secs  = qemu_opt_get_number(opts, "secs", 0);



    snapshot = qemu_opt_get_bool(opts, "snapshot", 0);

    ro = qemu_opt_get_bool(opts, "readonly", 0);



    file = qemu_opt_get(opts, "file");

    serial = qemu_opt_get(opts, "serial");



    if ((buf = qemu_opt_get(opts, "if")) != NULL) {

        pstrcpy(devname, sizeof(devname), buf);

        if (!strcmp(buf, "ide")) {

	    type = IF_IDE;

            max_devs = MAX_IDE_DEVS;

        } else if (!strcmp(buf, "scsi")) {

	    type = IF_SCSI;

            max_devs = MAX_SCSI_DEVS;

        } else if (!strcmp(buf, "floppy")) {

	    type = IF_FLOPPY;

            max_devs = 0;

        } else if (!strcmp(buf, "pflash")) {

	    type = IF_PFLASH;

            max_devs = 0;

	} else if (!strcmp(buf, "mtd")) {

	    type = IF_MTD;

            max_devs = 0;

	} else if (!strcmp(buf, "sd")) {

	    type = IF_SD;

            max_devs = 0;

        } else if (!strcmp(buf, "virtio")) {

            type = IF_VIRTIO;

            max_devs = 0;

	} else if (!strcmp(buf, "xen")) {

	    type = IF_XEN;

            max_devs = 0;

	} else if (!strcmp(buf, "none")) {

	    type = IF_NONE;

            max_devs = 0;

	} else {

            error_report("unsupported bus type '%s'", buf);

            return NULL;

	}

    }



    if (cyls || heads || secs) {

        if (cyls < 1 || (type == IF_IDE && cyls > 16383)) {

            error_report("invalid physical cyls number");

	    return NULL;

	}

        if (heads < 1 || (type == IF_IDE && heads > 16)) {

            error_report("invalid physical heads number");

	    return NULL;

	}

        if (secs < 1 || (type == IF_IDE && secs > 63)) {

            error_report("invalid physical secs number");

	    return NULL;

	}

    }



    if ((buf = qemu_opt_get(opts, "trans")) != NULL) {

        if (!cyls) {

            error_report("'%s' trans must be used with cyls,heads and secs",

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

                error_report("'%s' invalid physical CHS format", buf);

	        return NULL;

            }

	    media = MEDIA_CDROM;

	} else {

	    error_report("'%s' invalid media", buf);

	    return NULL;

	}

    }



    if ((buf = qemu_opt_get(opts, "cache")) != NULL) {

        if (!strcmp(buf, "off") || !strcmp(buf, "none")) {

            bdrv_flags |= BDRV_O_NOCACHE;

        } else if (!strcmp(buf, "writeback")) {

            bdrv_flags |= BDRV_O_CACHE_WB;

        } else if (!strcmp(buf, "unsafe")) {

            bdrv_flags |= BDRV_O_CACHE_WB;

            bdrv_flags |= BDRV_O_NO_FLUSH;

        } else if (!strcmp(buf, "writethrough")) {

            /* this is the default */

        } else {

           error_report("invalid cache option");

           return NULL;

        }

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

       if (strcmp(buf, "?") == 0) {

           error_printf("Supported formats:");

           bdrv_iterate_format(bdrv_format_print, NULL);

           error_printf("\n");

           return NULL;

        }

        drv = bdrv_find_whitelisted_format(buf);

        if (!drv) {

            error_report("'%s' invalid format", buf);

            return NULL;

        }

    }



    on_write_error = BLOCK_ERR_STOP_ENOSPC;

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



    on_read_error = BLOCK_ERR_REPORT;

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

        if (max_devs == 0)

        {

            unit_id = index;

            bus_id = 0;

        } else {

            unit_id = index % max_devs;

            bus_id = index / max_devs;

        }

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

     * ignore multiple definitions

     */



    if (drive_get(type, bus_id, unit_id) != NULL) {

        *fatal_error = 0;

        return NULL;

    }



    /* init */



    dinfo = qemu_mallocz(sizeof(*dinfo));

    if ((buf = qemu_opts_id(opts)) != NULL) {

        dinfo->id = qemu_strdup(buf);

    } else {

        /* no id supplied -> create one */

        dinfo->id = qemu_mallocz(32);

        if (type == IF_IDE || type == IF_SCSI)

            mediastr = (media == MEDIA_CDROM) ? "-cd" : "-hd";

        if (max_devs)

            snprintf(dinfo->id, 32, "%s%i%s%i",

                     devname, bus_id, mediastr, unit_id);

        else

            snprintf(dinfo->id, 32, "%s%s%i",

                     devname, mediastr, unit_id);

    }

    dinfo->bdrv = bdrv_new(dinfo->id);

    dinfo->devaddr = devaddr;

    dinfo->type = type;

    dinfo->bus = bus_id;

    dinfo->unit = unit_id;

    dinfo->opts = opts;

    if (serial)

        strncpy(dinfo->serial, serial, sizeof(dinfo->serial) - 1);

    QTAILQ_INSERT_TAIL(&drives, dinfo, next);



    bdrv_set_on_error(dinfo->bdrv, on_read_error, on_write_error);



    switch(type) {

    case IF_IDE:

    case IF_SCSI:

    case IF_XEN:

    case IF_NONE:

        switch(media) {

	case MEDIA_DISK:

            if (cyls != 0) {

                bdrv_set_geometry_hint(dinfo->bdrv, cyls, heads, secs);

                bdrv_set_translation_hint(dinfo->bdrv, translation);

            }

	    break;

	case MEDIA_CDROM:

            bdrv_set_type_hint(dinfo->bdrv, BDRV_TYPE_CDROM);

	    break;

	}

        break;

    case IF_SD:

        /* FIXME: This isn't really a floppy, but it's a reasonable

           approximation.  */

    case IF_FLOPPY:

        bdrv_set_type_hint(dinfo->bdrv, BDRV_TYPE_FLOPPY);

        break;

    case IF_PFLASH:

    case IF_MTD:

        break;

    case IF_VIRTIO:

        /* add virtio block device */

        opts = qemu_opts_create(qemu_find_opts("device"), NULL, 0);

        qemu_opt_set(opts, "driver", "virtio-blk-pci");

        qemu_opt_set(opts, "drive", dinfo->id);

        if (devaddr)

            qemu_opt_set(opts, "addr", devaddr);

        break;

    case IF_COUNT:

        abort();

    }

    if (!file || !*file) {

        *fatal_error = 0;

        return NULL;

    }

    if (snapshot) {

        /* always use cache=unsafe with snapshot */

        bdrv_flags &= ~BDRV_O_CACHE_MASK;

        bdrv_flags |= (BDRV_O_SNAPSHOT|BDRV_O_CACHE_WB|BDRV_O_NO_FLUSH);

    }



    if (media == MEDIA_CDROM) {

        /* CDROM is fine for any interface, don't check.  */

        ro = 1;

    } else if (ro == 1) {

        if (type != IF_SCSI && type != IF_VIRTIO && type != IF_FLOPPY && type != IF_NONE) {

            error_report("readonly not supported by this bus type");

            return NULL;

        }

    }



    bdrv_flags |= ro ? 0 : BDRV_O_RDWR;



    ret = bdrv_open(dinfo->bdrv, file, bdrv_flags, drv);

    if (ret < 0) {

        error_report("could not open disk image %s: %s",

                     file, strerror(-ret));

        return NULL;

    }



    if (bdrv_key_required(dinfo->bdrv))

        autostart = 0;

    *fatal_error = 0;

    return dinfo;

}
