static void bdrv_query_info(BlockBackend *blk, BlockInfo **p_info,

                            Error **errp)

{

    BlockInfo *info = g_malloc0(sizeof(*info));

    BlockDriverState *bs = blk_bs(blk);

    BlockDriverState *bs0;

    ImageInfo **p_image_info;

    Error *local_err = NULL;

    info->device = g_strdup(blk_name(blk));

    info->type = g_strdup("unknown");

    info->locked = blk_dev_is_medium_locked(blk);

    info->removable = blk_dev_has_removable_media(blk);



    if (blk_dev_has_removable_media(blk)) {

        info->has_tray_open = true;

        info->tray_open = blk_dev_is_tray_open(blk);

    }



    if (bdrv_iostatus_is_enabled(bs)) {

        info->has_io_status = true;

        info->io_status = bs->iostatus;

    }



    if (!QLIST_EMPTY(&bs->dirty_bitmaps)) {

        info->has_dirty_bitmaps = true;

        info->dirty_bitmaps = bdrv_query_dirty_bitmaps(bs);

    }



    if (bs->drv) {

        info->has_inserted = true;

        info->inserted = bdrv_block_device_info(bs);



        bs0 = bs;

        p_image_info = &info->inserted->image;

        while (1) {

            bdrv_query_image_info(bs0, p_image_info, &local_err);

            if (local_err) {

                error_propagate(errp, local_err);

                goto err;

            }

            if (bs0->drv && bs0->backing_hd) {

                bs0 = bs0->backing_hd;

                (*p_image_info)->has_backing_image = true;

                p_image_info = &((*p_image_info)->backing_image);

            } else {

                break;

            }

        }

    }



    *p_info = info;

    return;



 err:

    qapi_free_BlockInfo(info);

}
