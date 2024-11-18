BlockDeviceInfo *bdrv_block_device_info(BlockBackend *blk,

                                        BlockDriverState *bs, Error **errp)

{

    ImageInfo **p_image_info;

    BlockDriverState *bs0;

    BlockDeviceInfo *info = g_malloc0(sizeof(*info));



    info->file                   = g_strdup(bs->filename);

    info->ro                     = bs->read_only;

    info->drv                    = g_strdup(bs->drv->format_name);

    info->encrypted              = bs->encrypted;

    info->encryption_key_missing = false;



    info->cache = g_new(BlockdevCacheInfo, 1);

    *info->cache = (BlockdevCacheInfo) {

        .writeback      = blk ? blk_enable_write_cache(blk) : true,

        .direct         = !!(bs->open_flags & BDRV_O_NOCACHE),

        .no_flush       = !!(bs->open_flags & BDRV_O_NO_FLUSH),

    };



    if (bs->node_name[0]) {

        info->has_node_name = true;

        info->node_name = g_strdup(bs->node_name);

    }



    if (bs->backing_file[0]) {

        info->has_backing_file = true;

        info->backing_file = g_strdup(bs->backing_file);

    }



    info->detect_zeroes = bs->detect_zeroes;



    if (blk && blk_get_public(blk)->throttle_state) {

        ThrottleConfig cfg;



        throttle_group_get_config(blk, &cfg);



        info->bps     = cfg.buckets[THROTTLE_BPS_TOTAL].avg;

        info->bps_rd  = cfg.buckets[THROTTLE_BPS_READ].avg;

        info->bps_wr  = cfg.buckets[THROTTLE_BPS_WRITE].avg;



        info->iops    = cfg.buckets[THROTTLE_OPS_TOTAL].avg;

        info->iops_rd = cfg.buckets[THROTTLE_OPS_READ].avg;

        info->iops_wr = cfg.buckets[THROTTLE_OPS_WRITE].avg;



        info->has_bps_max     = cfg.buckets[THROTTLE_BPS_TOTAL].max;

        info->bps_max         = cfg.buckets[THROTTLE_BPS_TOTAL].max;

        info->has_bps_rd_max  = cfg.buckets[THROTTLE_BPS_READ].max;

        info->bps_rd_max      = cfg.buckets[THROTTLE_BPS_READ].max;

        info->has_bps_wr_max  = cfg.buckets[THROTTLE_BPS_WRITE].max;

        info->bps_wr_max      = cfg.buckets[THROTTLE_BPS_WRITE].max;



        info->has_iops_max    = cfg.buckets[THROTTLE_OPS_TOTAL].max;

        info->iops_max        = cfg.buckets[THROTTLE_OPS_TOTAL].max;

        info->has_iops_rd_max = cfg.buckets[THROTTLE_OPS_READ].max;

        info->iops_rd_max     = cfg.buckets[THROTTLE_OPS_READ].max;

        info->has_iops_wr_max = cfg.buckets[THROTTLE_OPS_WRITE].max;

        info->iops_wr_max     = cfg.buckets[THROTTLE_OPS_WRITE].max;



        info->has_bps_max_length     = info->has_bps_max;

        info->bps_max_length         =

            cfg.buckets[THROTTLE_BPS_TOTAL].burst_length;

        info->has_bps_rd_max_length  = info->has_bps_rd_max;

        info->bps_rd_max_length      =

            cfg.buckets[THROTTLE_BPS_READ].burst_length;

        info->has_bps_wr_max_length  = info->has_bps_wr_max;

        info->bps_wr_max_length      =

            cfg.buckets[THROTTLE_BPS_WRITE].burst_length;



        info->has_iops_max_length    = info->has_iops_max;

        info->iops_max_length        =

            cfg.buckets[THROTTLE_OPS_TOTAL].burst_length;

        info->has_iops_rd_max_length = info->has_iops_rd_max;

        info->iops_rd_max_length     =

            cfg.buckets[THROTTLE_OPS_READ].burst_length;

        info->has_iops_wr_max_length = info->has_iops_wr_max;

        info->iops_wr_max_length     =

            cfg.buckets[THROTTLE_OPS_WRITE].burst_length;



        info->has_iops_size = cfg.op_size;

        info->iops_size = cfg.op_size;



        info->has_group = true;

        info->group = g_strdup(throttle_group_get_name(blk));

    }



    info->write_threshold = bdrv_write_threshold_get(bs);



    bs0 = bs;

    p_image_info = &info->image;

    info->backing_file_depth = 0;

    while (1) {

        Error *local_err = NULL;

        bdrv_query_image_info(bs0, p_image_info, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            qapi_free_BlockDeviceInfo(info);

            return NULL;

        }



        if (bs0->drv && bs0->backing) {

            info->backing_file_depth++;

            bs0 = bs0->backing->bs;

            (*p_image_info)->has_backing_image = true;

            p_image_info = &((*p_image_info)->backing_image);

        } else {

            break;

        }



        /* Skip automatically inserted nodes that the user isn't aware of for

         * query-block (blk != NULL), but not for query-named-block-nodes */

        while (blk && bs0 && bs0->drv && bs0->implicit) {

            bs0 = backing_bs(bs0);

        }

    }



    return info;

}
