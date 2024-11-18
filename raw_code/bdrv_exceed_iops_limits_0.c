static bool bdrv_exceed_iops_limits(BlockDriverState *bs, bool is_write,

                             double elapsed_time, uint64_t *wait)

{

    uint64_t iops_limit = 0;

    double   ios_limit, ios_base;

    double   slice_time, wait_time;



    if (bs->io_limits.iops[BLOCK_IO_LIMIT_TOTAL]) {

        iops_limit = bs->io_limits.iops[BLOCK_IO_LIMIT_TOTAL];

    } else if (bs->io_limits.iops[is_write]) {

        iops_limit = bs->io_limits.iops[is_write];

    } else {

        if (wait) {

            *wait = 0;

        }



        return false;

    }



    slice_time = bs->slice_end - bs->slice_start;

    slice_time /= (NANOSECONDS_PER_SECOND);

    ios_limit  = iops_limit * slice_time;

    ios_base   = bs->nr_ops[is_write] - bs->io_base.ios[is_write];

    if (bs->io_limits.iops[BLOCK_IO_LIMIT_TOTAL]) {

        ios_base += bs->nr_ops[!is_write] - bs->io_base.ios[!is_write];

    }



    if (ios_base + 1 <= ios_limit) {

        if (wait) {

            *wait = 0;

        }



        return false;

    }



    /* Calc approx time to dispatch */

    wait_time = (ios_base + 1) / iops_limit;

    if (wait_time > elapsed_time) {

        wait_time = wait_time - elapsed_time;

    } else {

        wait_time = 0;

    }



    bs->slice_time = wait_time * BLOCK_IO_SLICE_TIME * 10;

    bs->slice_end += bs->slice_time - 3 * BLOCK_IO_SLICE_TIME;

    if (wait) {

        *wait = wait_time * BLOCK_IO_SLICE_TIME * 10;

    }



    return true;

}
