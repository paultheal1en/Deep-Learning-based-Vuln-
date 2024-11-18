void bdrv_close(BlockDriverState *bs)

{

    BdrvAioNotifier *ban, *ban_next;



    if (bs->job) {

        block_job_cancel_sync(bs->job);

    }



    /* Disable I/O limits and drain all pending throttled requests */

    if (bs->io_limits_enabled) {

        bdrv_io_limits_disable(bs);

    }



    bdrv_drain(bs); /* complete I/O */

    bdrv_flush(bs);

    bdrv_drain(bs); /* in case flush left pending I/O */

    notifier_list_notify(&bs->close_notifiers, bs);



    if (bs->drv) {

        BdrvChild *child, *next;



        bs->drv->bdrv_close(bs);

        bs->drv = NULL;



        bdrv_set_backing_hd(bs, NULL);



        if (bs->file != NULL) {

            bdrv_unref_child(bs, bs->file);

            bs->file = NULL;

        }



        QLIST_FOREACH_SAFE(child, &bs->children, next, next) {

            /* TODO Remove bdrv_unref() from drivers' close function and use

             * bdrv_unref_child() here */

            if (child->bs->inherits_from == bs) {

                child->bs->inherits_from = NULL;

            }

            bdrv_detach_child(child);

        }



        g_free(bs->opaque);

        bs->opaque = NULL;

        bs->copy_on_read = 0;

        bs->backing_file[0] = '\0';

        bs->backing_format[0] = '\0';

        bs->total_sectors = 0;

        bs->encrypted = 0;

        bs->valid_key = 0;

        bs->sg = 0;

        bs->zero_beyond_eof = false;

        QDECREF(bs->options);

        bs->options = NULL;

        QDECREF(bs->full_open_options);

        bs->full_open_options = NULL;

    }



    if (bs->blk) {

        blk_dev_change_media_cb(bs->blk, false);

    }



    QLIST_FOREACH_SAFE(ban, &bs->aio_notifiers, list, ban_next) {

        g_free(ban);

    }

    QLIST_INIT(&bs->aio_notifiers);

}
