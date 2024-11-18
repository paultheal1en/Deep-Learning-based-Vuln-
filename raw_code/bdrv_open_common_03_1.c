static int bdrv_open_common(BlockDriverState *bs, BlockDriverState *file,

    QDict *options, int flags, BlockDriver *drv)

{

    int ret, open_flags;

    const char *filename;



    assert(drv != NULL);

    assert(bs->file == NULL);

    assert(options != NULL && bs->options != options);



    trace_bdrv_open_common(bs, filename, flags, drv->format_name);



    if (use_bdrv_whitelist && !bdrv_is_whitelisted(drv)) {

        return -ENOTSUP;

    }



    /* bdrv_open() with directly using a protocol as drv. This layer is already

     * opened, so assign it to bs (while file becomes a closed BlockDriverState)

     * and return immediately. */

    if (file != NULL && drv->bdrv_file_open) {

        bdrv_swap(file, bs);

        return 0;

    }



    bs->open_flags = flags;

    bs->buffer_alignment = 512;



    assert(bs->copy_on_read == 0); /* bdrv_new() and bdrv_close() make it so */

    if ((flags & BDRV_O_RDWR) && (flags & BDRV_O_COPY_ON_READ)) {

        bdrv_enable_copy_on_read(bs);

    }



    if (file != NULL) {

        filename = file->filename;

    } else {

        filename = qdict_get_try_str(options, "filename");

    }



    if (filename != NULL) {

        pstrcpy(bs->filename, sizeof(bs->filename), filename);

    } else {

        bs->filename[0] = '\0';

    }



    bs->drv = drv;

    bs->opaque = g_malloc0(drv->instance_size);



    bs->enable_write_cache = !!(flags & BDRV_O_CACHE_WB);

    open_flags = bdrv_open_flags(bs, flags);



    bs->read_only = !(open_flags & BDRV_O_RDWR);



    /* Open the image, either directly or using a protocol */

    if (drv->bdrv_file_open) {

        assert(file == NULL);

        assert(drv->bdrv_parse_filename || filename != NULL);

        ret = drv->bdrv_file_open(bs, options, open_flags);

    } else {

        if (file == NULL) {

            qerror_report(ERROR_CLASS_GENERIC_ERROR, "Can't use '%s' as a "

                          "block driver for the protocol level",

                          drv->format_name);

            ret = -EINVAL;

            goto free_and_fail;

        }

        assert(file != NULL);

        bs->file = file;

        ret = drv->bdrv_open(bs, options, open_flags);

    }



    if (ret < 0) {

        goto free_and_fail;

    }



    ret = refresh_total_sectors(bs, bs->total_sectors);

    if (ret < 0) {

        goto free_and_fail;

    }



#ifndef _WIN32

    if (bs->is_temporary) {

        assert(filename != NULL);

        unlink(filename);

    }

#endif

    return 0;



free_and_fail:

    bs->file = NULL;

    g_free(bs->opaque);

    bs->opaque = NULL;

    bs->drv = NULL;

    return ret;

}
