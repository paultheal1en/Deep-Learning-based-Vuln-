static bool bdrv_exceed_bps_limits(BlockDriverState *bs, int nb_sectors,

                 bool is_write, double elapsed_time, uint64_t *wait)

{

    uint64_t bps_limit = 0;

    double   bytes_limit, bytes_base, bytes_res;

    double   slice_time, wait_time;



    if (bs->io_limits.bps[BLOCK_IO_LIMIT_TOTAL]) {

        bps_limit = bs->io_limits.bps[BLOCK_IO_LIMIT_TOTAL];

    } else if (bs->io_limits.bps[is_write]) {

        bps_limit = bs->io_limits.bps[is_write];

    } else {

        if (wait) {

            *wait = 0;

        }



        return false;

    }



    slice_time = bs->slice_end - bs->slice_start;

    slice_time /= (NANOSECONDS_PER_SECOND);

    bytes_limit = bps_limit * slice_time;

    bytes_base  = bs->nr_bytes[is_write] - bs->io_base.bytes[is_write];

    if (bs->io_limits.bps[BLOCK_IO_LIMIT_TOTAL]) {

        bytes_base += bs->nr_bytes[!is_write] - bs->io_base.bytes[!is_write];

    }



    /* bytes_base: the bytes of data which have been read/written; and

     *             it is obtained from the history statistic info.

     * bytes_res: the remaining bytes of data which need to be read/written.

     * (bytes_base + bytes_res) / bps_limit: used to calcuate

     *             the total time for completing reading/writting all data.

     */

    bytes_res   = (unsigned) nb_sectors * BDRV_SECTOR_SIZE;



    if (bytes_base + bytes_res <= bytes_limit) {

        if (wait) {

            *wait = 0;

        }



        return false;

    }



    /* Calc approx time to dispatch */

    wait_time = (bytes_base + bytes_res) / bps_limit - elapsed_time;



    /* When the I/O rate at runtime exceeds the limits,

     * bs->slice_end need to be extended in order that the current statistic

     * info can be kept until the timer fire, so it is increased and tuned

     * based on the result of experiment.

     */

    bs->slice_time = wait_time * BLOCK_IO_SLICE_TIME * 10;

    bs->slice_end += bs->slice_time - 3 * BLOCK_IO_SLICE_TIME;

    if (wait) {

        *wait = wait_time * BLOCK_IO_SLICE_TIME * 10;

    }



    return true;

}
