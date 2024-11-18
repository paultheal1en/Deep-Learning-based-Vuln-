void qmp_block_set_io_throttle(const char *device, int64_t bps, int64_t bps_rd,

                               int64_t bps_wr, int64_t iops, int64_t iops_rd,

                               int64_t iops_wr, Error **errp)

{

    BlockIOLimit io_limits;

    BlockDriverState *bs;



    bs = bdrv_find(device);

    if (!bs) {

        error_set(errp, QERR_DEVICE_NOT_FOUND, device);

        return;

    }



    io_limits.bps[BLOCK_IO_LIMIT_TOTAL] = bps;

    io_limits.bps[BLOCK_IO_LIMIT_READ]  = bps_rd;

    io_limits.bps[BLOCK_IO_LIMIT_WRITE] = bps_wr;

    io_limits.iops[BLOCK_IO_LIMIT_TOTAL]= iops;

    io_limits.iops[BLOCK_IO_LIMIT_READ] = iops_rd;

    io_limits.iops[BLOCK_IO_LIMIT_WRITE]= iops_wr;



    if (!do_check_io_limits(&io_limits)) {

        error_set(errp, QERR_INVALID_PARAMETER_COMBINATION);

        return;

    }



    bs->io_limits = io_limits;

    bs->slice_time = BLOCK_IO_SLICE_TIME;



    if (!bs->io_limits_enabled && bdrv_io_limits_enabled(bs)) {

        bdrv_io_limits_enable(bs);

    } else if (bs->io_limits_enabled && !bdrv_io_limits_enabled(bs)) {

        bdrv_io_limits_disable(bs);

    } else {

        if (bs->block_timer) {

            qemu_mod_timer(bs->block_timer, qemu_get_clock_ns(vm_clock));

        }

    }

}
