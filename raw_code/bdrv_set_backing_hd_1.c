void bdrv_set_backing_hd(BlockDriverState *bs, BlockDriverState *backing_hd)

{



    if (bs->backing_hd) {

        assert(bs->backing_blocker);

        bdrv_op_unblock_all(bs->backing_hd, bs->backing_blocker);

    } else if (backing_hd) {

        error_setg(&bs->backing_blocker,

                   "device is used as backing hd of '%s'",

                   bdrv_get_device_name(bs));

    }



    bs->backing_hd = backing_hd;

    if (!backing_hd) {

        error_free(bs->backing_blocker);

        bs->backing_blocker = NULL;

        goto out;

    }

    bs->open_flags &= ~BDRV_O_NO_BACKING;

    pstrcpy(bs->backing_file, sizeof(bs->backing_file), backing_hd->filename);

    pstrcpy(bs->backing_format, sizeof(bs->backing_format),

            backing_hd->drv ? backing_hd->drv->format_name : "");



    bdrv_op_block_all(bs->backing_hd, bs->backing_blocker);

    /* Otherwise we won't be able to commit due to check in bdrv_commit */

    bdrv_op_unblock(bs->backing_hd, BLOCK_OP_TYPE_COMMIT,

                    bs->backing_blocker);

out:

    bdrv_refresh_limits(bs, NULL);

}
