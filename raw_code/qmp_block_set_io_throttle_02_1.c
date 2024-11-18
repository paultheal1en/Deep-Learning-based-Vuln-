void qmp_block_set_io_throttle(const char *device, int64_t bps, int64_t bps_rd,

                               int64_t bps_wr, int64_t iops, int64_t iops_rd,

                               int64_t iops_wr, Error **errp)

{

    ThrottleConfig cfg;

    BlockDriverState *bs;



    bs = bdrv_find(device);

    if (!bs) {

        error_set(errp, QERR_DEVICE_NOT_FOUND, device);

        return;

    }



    memset(&cfg, 0, sizeof(cfg));

    cfg.buckets[THROTTLE_BPS_TOTAL].avg = bps;

    cfg.buckets[THROTTLE_BPS_READ].avg  = bps_rd;

    cfg.buckets[THROTTLE_BPS_WRITE].avg = bps_wr;



    cfg.buckets[THROTTLE_OPS_TOTAL].avg = iops;

    cfg.buckets[THROTTLE_OPS_READ].avg  = iops_rd;

    cfg.buckets[THROTTLE_OPS_WRITE].avg = iops_wr;



    cfg.buckets[THROTTLE_BPS_TOTAL].max = 0;

    cfg.buckets[THROTTLE_BPS_READ].max  = 0;

    cfg.buckets[THROTTLE_BPS_WRITE].max = 0;



    cfg.buckets[THROTTLE_OPS_TOTAL].max = 0;

    cfg.buckets[THROTTLE_OPS_READ].max  = 0;

    cfg.buckets[THROTTLE_OPS_WRITE].max = 0;



    cfg.op_size = 0;



    if (!check_throttle_config(&cfg, errp)) {

        return;

    }



    if (!bs->io_limits_enabled && throttle_enabled(&cfg)) {

        bdrv_io_limits_enable(bs);

    } else if (bs->io_limits_enabled && !throttle_enabled(&cfg)) {

        bdrv_io_limits_disable(bs);

    }



    if (bs->io_limits_enabled) {

        bdrv_set_io_limits(bs, &cfg);

    }

}
