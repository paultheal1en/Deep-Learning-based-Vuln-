int drive_init(struct drive_opt *arg, int snapshot, void *opaque)

{

    char buf[128];

    char file[1024];

    char devname[128];

    char serial[21];

    const char *mediastr = "";

    BlockInterfaceType type;

    enum { MEDIA_DISK, MEDIA_CDROM } media;

    int bus_id, unit_id;

    int cyls, heads, secs, translation;

    BlockDriverState *bdrv;

    BlockDriver *drv = NULL;

    QEMUMachine *machine = opaque;

    int max_devs;

    int index;

    int cache;

    int bdrv_flags, onerror;

    int drives_table_idx;

    char *str = arg->opt;

    static const char * const params[] = { "bus", "unit", "if", "index",

                                           "cyls", "heads", "secs", "trans",

                                           "media", "snapshot", "file",

                                           "cache", "format", "serial", "werror",

                                           NULL };



    if (check_params(params, str) < 0) {

         fprintf(stderr, "qemu: unknown parameter '%s' in '%s'\n",

                         buf, str);

         return -1;

    }



    file[0] = 0;

    cyls = heads = secs = 0;

    bus_id = 0;

    unit_id = -1;

    translation = BIOS_ATA_TRANSLATION_AUTO;

    index = -1;

    cache = 3;



    if (machine->use_scsi) {

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



    if (get_param_value(buf, sizeof(buf), "bus", str)) {

        bus_id = strtol(buf, NULL, 0);

	if (bus_id < 0) {

	    fprintf(stderr, "qemu: '%s' invalid bus id\n", str);

	    return -1;

	}

    }



    if (get_param_value(buf, sizeof(buf), "unit", str)) {

        unit_id = strtol(buf, NULL, 0);

	if (unit_id < 0) {

	    fprintf(stderr, "qemu: '%s' invalid unit id\n", str);

	    return -1;

	}

    }



    if (get_param_value(buf, sizeof(buf), "if", str)) {

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

	} else {

            fprintf(stderr, "qemu: '%s' unsupported bus type '%s'\n", str, buf);

            return -1;

	}

    }



    if (get_param_value(buf, sizeof(buf), "index", str)) {

        index = strtol(buf, NULL, 0);

	if (index < 0) {

	    fprintf(stderr, "qemu: '%s' invalid index\n", str);

	    return -1;

	}

    }



    if (get_param_value(buf, sizeof(buf), "cyls", str)) {

        cyls = strtol(buf, NULL, 0);

    }



    if (get_param_value(buf, sizeof(buf), "heads", str)) {

        heads = strtol(buf, NULL, 0);

    }



    if (get_param_value(buf, sizeof(buf), "secs", str)) {

        secs = strtol(buf, NULL, 0);

    }



    if (cyls || heads || secs) {

        if (cyls < 1 || cyls > 16383) {

            fprintf(stderr, "qemu: '%s' invalid physical cyls number\n", str);

	    return -1;

	}

        if (heads < 1 || heads > 16) {

            fprintf(stderr, "qemu: '%s' invalid physical heads number\n", str);

	    return -1;

	}

        if (secs < 1 || secs > 63) {

            fprintf(stderr, "qemu: '%s' invalid physical secs number\n", str);

	    return -1;

	}

    }



    if (get_param_value(buf, sizeof(buf), "trans", str)) {

        if (!cyls) {

            fprintf(stderr,

                    "qemu: '%s' trans must be used with cyls,heads and secs\n",

                    str);

            return -1;

        }

        if (!strcmp(buf, "none"))

            translation = BIOS_ATA_TRANSLATION_NONE;

        else if (!strcmp(buf, "lba"))

            translation = BIOS_ATA_TRANSLATION_LBA;

        else if (!strcmp(buf, "auto"))

            translation = BIOS_ATA_TRANSLATION_AUTO;

	else {

            fprintf(stderr, "qemu: '%s' invalid translation type\n", str);

	    return -1;

	}

    }



    if (get_param_value(buf, sizeof(buf), "media", str)) {

        if (!strcmp(buf, "disk")) {

	    media = MEDIA_DISK;

	} else if (!strcmp(buf, "cdrom")) {

            if (cyls || secs || heads) {

                fprintf(stderr,

                        "qemu: '%s' invalid physical CHS format\n", str);

	        return -1;

            }

	    media = MEDIA_CDROM;

	} else {

	    fprintf(stderr, "qemu: '%s' invalid media\n", str);

	    return -1;

	}

    }



    if (get_param_value(buf, sizeof(buf), "snapshot", str)) {

        if (!strcmp(buf, "on"))

	    snapshot = 1;

        else if (!strcmp(buf, "off"))

	    snapshot = 0;

	else {

	    fprintf(stderr, "qemu: '%s' invalid snapshot option\n", str);

	    return -1;

	}

    }



    if (get_param_value(buf, sizeof(buf), "cache", str)) {

        if (!strcmp(buf, "off") || !strcmp(buf, "none"))

            cache = 0;

        else if (!strcmp(buf, "writethrough"))

            cache = 1;

        else if (!strcmp(buf, "writeback"))

            cache = 2;

        else {

           fprintf(stderr, "qemu: invalid cache option\n");

           return -1;

        }

    }



    if (get_param_value(buf, sizeof(buf), "format", str)) {

       if (strcmp(buf, "?") == 0) {

            fprintf(stderr, "qemu: Supported formats:");

            bdrv_iterate_format(bdrv_format_print, NULL);

            fprintf(stderr, "\n");

	    return -1;

        }

        drv = bdrv_find_format(buf);

        if (!drv) {

            fprintf(stderr, "qemu: '%s' invalid format\n", buf);

            return -1;

        }

    }



    if (arg->file == NULL)

        get_param_value(file, sizeof(file), "file", str);

    else

        pstrcpy(file, sizeof(file), arg->file);



    if (!get_param_value(serial, sizeof(serial), "serial", str))

	    memset(serial, 0,  sizeof(serial));



    onerror = BLOCK_ERR_STOP_ENOSPC;

    if (get_param_value(buf, sizeof(serial), "werror", str)) {

        if (type != IF_IDE && type != IF_SCSI && type != IF_VIRTIO) {

            fprintf(stderr, "werror is no supported by this format\n");

            return -1;

        }

        if (!strcmp(buf, "ignore"))

            onerror = BLOCK_ERR_IGNORE;

        else if (!strcmp(buf, "enospc"))

            onerror = BLOCK_ERR_STOP_ENOSPC;

        else if (!strcmp(buf, "stop"))

            onerror = BLOCK_ERR_STOP_ANY;

        else if (!strcmp(buf, "report"))

            onerror = BLOCK_ERR_REPORT;

        else {

            fprintf(stderr, "qemu: '%s' invalid write error action\n", buf);

            return -1;

        }

    }



    /* compute bus and unit according index */



    if (index != -1) {

        if (bus_id != 0 || unit_id != -1) {

            fprintf(stderr,

                    "qemu: '%s' index cannot be used with bus and unit\n", str);

            return -1;

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

       while (drive_get_index(type, bus_id, unit_id) != -1) {

           unit_id++;

           if (max_devs && unit_id >= max_devs) {

               unit_id -= max_devs;

               bus_id++;

           }

       }

    }



    /* check unit id */



    if (max_devs && unit_id >= max_devs) {

        fprintf(stderr, "qemu: '%s' unit %d too big (max is %d)\n",

                        str, unit_id, max_devs - 1);

        return -1;

    }



    /*

     * ignore multiple definitions

     */



    if (drive_get_index(type, bus_id, unit_id) != -1)

        return -2;



    /* init */



    if (type == IF_IDE || type == IF_SCSI)

        mediastr = (media == MEDIA_CDROM) ? "-cd" : "-hd";

    if (max_devs)

        snprintf(buf, sizeof(buf), "%s%i%s%i",

                 devname, bus_id, mediastr, unit_id);

    else

        snprintf(buf, sizeof(buf), "%s%s%i",

                 devname, mediastr, unit_id);

    bdrv = bdrv_new(buf);

    drives_table_idx = drive_get_free_idx();

    drives_table[drives_table_idx].bdrv = bdrv;

    drives_table[drives_table_idx].type = type;

    drives_table[drives_table_idx].bus = bus_id;

    drives_table[drives_table_idx].unit = unit_id;

    drives_table[drives_table_idx].onerror = onerror;

    drives_table[drives_table_idx].drive_opt_idx = arg - drives_opt;

    strncpy(drives_table[drives_table_idx].serial, serial, sizeof(serial));

    nb_drives++;



    switch(type) {

    case IF_IDE:

    case IF_SCSI:

    case IF_XEN:

        switch(media) {

	case MEDIA_DISK:

            if (cyls != 0) {

                bdrv_set_geometry_hint(bdrv, cyls, heads, secs);

                bdrv_set_translation_hint(bdrv, translation);

            }

	    break;

	case MEDIA_CDROM:

            bdrv_set_type_hint(bdrv, BDRV_TYPE_CDROM);

	    break;

	}

        break;

    case IF_SD:

        /* FIXME: This isn't really a floppy, but it's a reasonable

           approximation.  */

    case IF_FLOPPY:

        bdrv_set_type_hint(bdrv, BDRV_TYPE_FLOPPY);

        break;

    case IF_PFLASH:

    case IF_MTD:

    case IF_VIRTIO:

        break;

    case IF_COUNT:

        abort();

    }

    if (!file[0])

        return -2;

    bdrv_flags = 0;

    if (snapshot) {

        bdrv_flags |= BDRV_O_SNAPSHOT;

        cache = 2; /* always use write-back with snapshot */

    }

    if (cache == 0) /* no caching */

        bdrv_flags |= BDRV_O_NOCACHE;

    else if (cache == 2) /* write-back */

        bdrv_flags |= BDRV_O_CACHE_WB;

    else if (cache == 3) /* not specified */

        bdrv_flags |= BDRV_O_CACHE_DEF;

    if (bdrv_open2(bdrv, file, bdrv_flags, drv) < 0) {

        fprintf(stderr, "qemu: could not open disk image %s\n",

                        file);

        return -1;

    }

    if (bdrv_key_required(bdrv))

        autostart = 0;

    return drives_table_idx;

}
