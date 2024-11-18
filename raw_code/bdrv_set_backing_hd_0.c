void bdrv_set_backing_hd(BlockDriverState *bs, BlockDriverState *backing_hd)

{

    if (backing_hd) {

        bdrv_ref(backing_hd);

    }



    if (bs->backing) {

        assert(bs->backing_blocker);

        bdrv_op_unblock_all(bs->backing->bs, bs->backing_blocker);

        bdrv_unref_child(bs, bs->backing);

    } else if (backing_hd) {

        error_setg(&bs->backing_blocker,

                   "node is used as backing hd of '%s'",

                   bdrv_get_device_or_node_name(bs));

    }



    if (!backing_hd) {

        error_free(bs->backing_blocker);

        bs->backing_blocker = NULL;

        bs->backing = NULL;

        goto out;

    }

    bs->backing = bdrv_attach_child(bs, backing_hd, &child_backing);

    bs->open_flags &= ~BDRV_O_NO_BACKING;

    pstrcpy(bs->backing_file, sizeof(bs->backing_file), backing_hd->filename);

    pstrcpy(bs->backing_format, sizeof(bs->backing_format),

            backing_hd->drv ? backing_hd->drv->format_name : "");



    bdrv_op_block_all(backing_hd, bs->backing_blocker);

    /* Otherwise we won't be able to commit due to check in bdrv_commit */

    bdrv_op_unblock(backing_hd, BLOCK_OP_TYPE_COMMIT_TARGET,

                    bs->backing_blocker);

out:

    bdrv_refresh_limits(bs, NULL);

}
