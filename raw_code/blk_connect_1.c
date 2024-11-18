static int blk_connect(struct XenDevice *xendev)

{

    struct XenBlkDev *blkdev = container_of(xendev, struct XenBlkDev, xendev);

    int pers, index, qflags;

    bool readonly = true;



    /* read-only ? */

    if (blkdev->directiosafe) {

        qflags = BDRV_O_NOCACHE | BDRV_O_NATIVE_AIO;

    } else {

        qflags = BDRV_O_CACHE_WB;

    }

    if (strcmp(blkdev->mode, "w") == 0) {

        qflags |= BDRV_O_RDWR;

        readonly = false;

    }

    if (blkdev->feature_discard) {

        qflags |= BDRV_O_UNMAP;

    }



    /* init qemu block driver */

    index = (blkdev->xendev.dev - 202 * 256) / 16;

    blkdev->dinfo = drive_get(IF_XEN, 0, index);

    if (!blkdev->dinfo) {

        Error *local_err = NULL;

        /* setup via xenbus -> create new block driver instance */

        xen_be_printf(&blkdev->xendev, 2, "create new bdrv (xenbus setup)\n");

        blkdev->bs = bdrv_new(blkdev->dev, &local_err);

        if (local_err) {

            blkdev->bs = NULL;

        }

        if (blkdev->bs) {

            BlockDriver *drv = bdrv_find_whitelisted_format(blkdev->fileproto,

                                                           readonly);

            if (bdrv_open(&blkdev->bs, blkdev->filename, NULL, NULL, qflags,

                          drv, &local_err) != 0)

            {

                xen_be_printf(&blkdev->xendev, 0, "error: %s\n",

                              error_get_pretty(local_err));

                error_free(local_err);

                bdrv_unref(blkdev->bs);

                blkdev->bs = NULL;

            }

        }

        if (!blkdev->bs) {

            return -1;

        }

    } else {

        /* setup via qemu cmdline -> already setup for us */

        xen_be_printf(&blkdev->xendev, 2, "get configured bdrv (cmdline setup)\n");

        blkdev->bs = blkdev->dinfo->bdrv;

        if (bdrv_is_read_only(blkdev->bs) && !readonly) {

            xen_be_printf(&blkdev->xendev, 0, "Unexpected read-only drive");

            blkdev->bs = NULL;

            return -1;

        }

        /* blkdev->bs is not create by us, we get a reference

         * so we can bdrv_unref() unconditionally */

        bdrv_ref(blkdev->bs);

    }

    bdrv_attach_dev_nofail(blkdev->bs, blkdev);

    blkdev->file_size = bdrv_getlength(blkdev->bs);

    if (blkdev->file_size < 0) {

        xen_be_printf(&blkdev->xendev, 1, "bdrv_getlength: %d (%s) | drv %s\n",

                      (int)blkdev->file_size, strerror(-blkdev->file_size),

                      bdrv_get_format_name(blkdev->bs) ?: "-");

        blkdev->file_size = 0;

    }



    xen_be_printf(xendev, 1, "type \"%s\", fileproto \"%s\", filename \"%s\","

                  " size %" PRId64 " (%" PRId64 " MB)\n",

                  blkdev->type, blkdev->fileproto, blkdev->filename,

                  blkdev->file_size, blkdev->file_size >> 20);



    /* Fill in number of sector size and number of sectors */

    xenstore_write_be_int(&blkdev->xendev, "sector-size", blkdev->file_blk);

    xenstore_write_be_int64(&blkdev->xendev, "sectors",

                            blkdev->file_size / blkdev->file_blk);



    if (xenstore_read_fe_int(&blkdev->xendev, "ring-ref", &blkdev->ring_ref) == -1) {

        return -1;

    }

    if (xenstore_read_fe_int(&blkdev->xendev, "event-channel",

                             &blkdev->xendev.remote_port) == -1) {

        return -1;

    }

    if (xenstore_read_fe_int(&blkdev->xendev, "feature-persistent", &pers)) {

        blkdev->feature_persistent = FALSE;

    } else {

        blkdev->feature_persistent = !!pers;

    }



    blkdev->protocol = BLKIF_PROTOCOL_NATIVE;

    if (blkdev->xendev.protocol) {

        if (strcmp(blkdev->xendev.protocol, XEN_IO_PROTO_ABI_X86_32) == 0) {

            blkdev->protocol = BLKIF_PROTOCOL_X86_32;

        }

        if (strcmp(blkdev->xendev.protocol, XEN_IO_PROTO_ABI_X86_64) == 0) {

            blkdev->protocol = BLKIF_PROTOCOL_X86_64;

        }

    }



    blkdev->sring = xc_gnttab_map_grant_ref(blkdev->xendev.gnttabdev,

                                            blkdev->xendev.dom,

                                            blkdev->ring_ref,

                                            PROT_READ | PROT_WRITE);

    if (!blkdev->sring) {

        return -1;

    }

    blkdev->cnt_map++;



    switch (blkdev->protocol) {

    case BLKIF_PROTOCOL_NATIVE:

    {

        blkif_sring_t *sring_native = blkdev->sring;

        BACK_RING_INIT(&blkdev->rings.native, sring_native, XC_PAGE_SIZE);

        break;

    }

    case BLKIF_PROTOCOL_X86_32:

    {

        blkif_x86_32_sring_t *sring_x86_32 = blkdev->sring;



        BACK_RING_INIT(&blkdev->rings.x86_32_part, sring_x86_32, XC_PAGE_SIZE);

        break;

    }

    case BLKIF_PROTOCOL_X86_64:

    {

        blkif_x86_64_sring_t *sring_x86_64 = blkdev->sring;



        BACK_RING_INIT(&blkdev->rings.x86_64_part, sring_x86_64, XC_PAGE_SIZE);

        break;

    }

    }



    if (blkdev->feature_persistent) {

        /* Init persistent grants */

        blkdev->max_grants = max_requests * BLKIF_MAX_SEGMENTS_PER_REQUEST;

        blkdev->persistent_gnts = g_tree_new_full((GCompareDataFunc)int_cmp,

                                             NULL, NULL,

                                             (GDestroyNotify)destroy_grant);

        blkdev->persistent_gnt_count = 0;

    }



    xen_be_bind_evtchn(&blkdev->xendev);



    xen_be_printf(&blkdev->xendev, 1, "ok: proto %s, ring-ref %d, "

                  "remote port %d, local port %d\n",

                  blkdev->xendev.protocol, blkdev->ring_ref,

                  blkdev->xendev.remote_port, blkdev->xendev.local_port);

    return 0;

}
