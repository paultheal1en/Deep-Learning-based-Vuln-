int bdrv_snapshot_goto(BlockDriverState *bs,

                       const char *snapshot_id,

                       Error **errp)

{

    BlockDriver *drv = bs->drv;

    int ret, open_ret;

    int64_t len;



    if (!drv) {

        error_setg(errp, "Block driver is closed");

        return -ENOMEDIUM;

    }



    len = bdrv_getlength(bs);

    if (len < 0) {

        error_setg_errno(errp, -len, "Cannot get block device size");

        return len;

    }

    /* We should set all bits in all enabled dirty bitmaps, because dirty

     * bitmaps reflect active state of disk and snapshot switch operation

     * actually dirties active state.

     * TODO: It may make sense not to set all bits but analyze block status of

     * current state and destination snapshot and do not set bits corresponding

     * to both-zero or both-unallocated areas. */

    bdrv_set_dirty(bs, 0, len);



    if (drv->bdrv_snapshot_goto) {

        ret = drv->bdrv_snapshot_goto(bs, snapshot_id);

        if (ret < 0) {

            error_setg_errno(errp, -ret, "Failed to load snapshot");

        }

        return ret;

    }



    if (bs->file) {

        BlockDriverState *file;

        QDict *options = qdict_clone_shallow(bs->options);

        QDict *file_options;

        Error *local_err = NULL;



        file = bs->file->bs;

        /* Prevent it from getting deleted when detached from bs */

        bdrv_ref(file);



        qdict_extract_subqdict(options, &file_options, "file.");

        QDECREF(file_options);

        qdict_put_str(options, "file", bdrv_get_node_name(file));



        drv->bdrv_close(bs);

        bdrv_unref_child(bs, bs->file);

        bs->file = NULL;



        ret = bdrv_snapshot_goto(file, snapshot_id, errp);

        open_ret = drv->bdrv_open(bs, options, bs->open_flags, &local_err);

        QDECREF(options);

        if (open_ret < 0) {

            bdrv_unref(file);

            bs->drv = NULL;

            /* A bdrv_snapshot_goto() error takes precedence */

            error_propagate(errp, local_err);

            return ret < 0 ? ret : open_ret;

        }



        assert(bs->file->bs == file);

        bdrv_unref(file);

        return ret;

    }



    error_setg(errp, "Block driver does not support snapshots");

    return -ENOTSUP;

}
