int bdrv_open(BlockDriverState *bs, const char *filename, int flags,

              BlockDriver *drv)

{

    int ret, open_flags;

    char tmp_filename[PATH_MAX];

    char backing_filename[PATH_MAX];



    bs->is_temporary = 0;

    bs->encrypted = 0;

    bs->valid_key = 0;

    bs->open_flags = flags;

    /* buffer_alignment defaulted to 512, drivers can change this value */

    bs->buffer_alignment = 512;



    if (flags & BDRV_O_SNAPSHOT) {

        BlockDriverState *bs1;

        int64_t total_size;

        int is_protocol = 0;

        BlockDriver *bdrv_qcow2;

        QEMUOptionParameter *options;



        /* if snapshot, we create a temporary backing file and open it

           instead of opening 'filename' directly */



        /* if there is a backing file, use it */

        bs1 = bdrv_new("");

        ret = bdrv_open(bs1, filename, 0, drv);

        if (ret < 0) {

            bdrv_delete(bs1);

            return ret;

        }

        total_size = bdrv_getlength(bs1) >> BDRV_SECTOR_BITS;



        if (bs1->drv && bs1->drv->protocol_name)

            is_protocol = 1;



        bdrv_delete(bs1);



        get_tmp_filename(tmp_filename, sizeof(tmp_filename));



        /* Real path is meaningless for protocols */

        if (is_protocol)

            snprintf(backing_filename, sizeof(backing_filename),

                     "%s", filename);

        else if (!realpath(filename, backing_filename))

            return -errno;



        bdrv_qcow2 = bdrv_find_format("qcow2");

        options = parse_option_parameters("", bdrv_qcow2->create_options, NULL);



        set_option_parameter_int(options, BLOCK_OPT_SIZE, total_size * 512);

        set_option_parameter(options, BLOCK_OPT_BACKING_FILE, backing_filename);

        if (drv) {

            set_option_parameter(options, BLOCK_OPT_BACKING_FMT,

                drv->format_name);

        }



        ret = bdrv_create(bdrv_qcow2, tmp_filename, options);

        if (ret < 0) {

            return ret;

        }



        filename = tmp_filename;

        drv = bdrv_qcow2;

        bs->is_temporary = 1;

    }



    pstrcpy(bs->filename, sizeof(bs->filename), filename);



    if (!drv) {

        drv = find_hdev_driver(filename);

        if (!drv) {

            drv = find_image_format(filename);

        }

    }



    if (!drv) {

        ret = -ENOENT;

        goto unlink_and_fail;

    }

    if (use_bdrv_whitelist && !bdrv_is_whitelisted(drv)) {

        ret = -ENOTSUP;

        goto unlink_and_fail;

    }



    bs->drv = drv;

    bs->opaque = qemu_mallocz(drv->instance_size);



    /*

     * Yes, BDRV_O_NOCACHE aka O_DIRECT means we have to present a

     * write cache to the guest.  We do need the fdatasync to flush

     * out transactions for block allocations, and we maybe have a

     * volatile write cache in our backing device to deal with.

     */

    if (flags & (BDRV_O_CACHE_WB|BDRV_O_NOCACHE))

        bs->enable_write_cache = 1;



    /*

     * Clear flags that are internal to the block layer before opening the

     * image.

     */

    open_flags = flags & ~(BDRV_O_SNAPSHOT | BDRV_O_NO_BACKING);



    /*

     * Snapshots should be writeable.

     */

    if (bs->is_temporary) {

        open_flags |= BDRV_O_RDWR;

    }



    ret = drv->bdrv_open(bs, filename, open_flags);

    if (ret < 0) {

        goto free_and_fail;

    }



    bs->keep_read_only = bs->read_only = !(open_flags & BDRV_O_RDWR);

    if (drv->bdrv_getlength) {

        bs->total_sectors = bdrv_getlength(bs) >> BDRV_SECTOR_BITS;

    }

#ifndef _WIN32

    if (bs->is_temporary) {

        unlink(filename);

    }

#endif

    if ((flags & BDRV_O_NO_BACKING) == 0 && bs->backing_file[0] != '\0') {

        /* if there is a backing file, use it */

        BlockDriver *back_drv = NULL;

        bs->backing_hd = bdrv_new("");

        path_combine(backing_filename, sizeof(backing_filename),

                     filename, bs->backing_file);

        if (bs->backing_format[0] != '\0')

            back_drv = bdrv_find_format(bs->backing_format);



        /* backing files always opened read-only */

        open_flags &= ~BDRV_O_RDWR;



        ret = bdrv_open(bs->backing_hd, backing_filename, open_flags, back_drv);

        if (ret < 0) {

            bdrv_close(bs);

            return ret;

        }

        if (bs->is_temporary) {

            bs->backing_hd->keep_read_only = !(flags & BDRV_O_RDWR);

        } else {

            /* base image inherits from "parent" */

            bs->backing_hd->keep_read_only = bs->keep_read_only;

        }

    }



    if (!bdrv_key_required(bs)) {

        /* call the change callback */

        bs->media_changed = 1;

        if (bs->change_cb)

            bs->change_cb(bs->change_opaque);

    }

    return 0;



free_and_fail:

    qemu_free(bs->opaque);

    bs->opaque = NULL;

    bs->drv = NULL;

unlink_and_fail:

    if (bs->is_temporary)

        unlink(filename);

    return ret;

}
