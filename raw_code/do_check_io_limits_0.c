static bool do_check_io_limits(BlockIOLimit *io_limits)

{

    bool bps_flag;

    bool iops_flag;



    assert(io_limits);



    bps_flag  = (io_limits->bps[BLOCK_IO_LIMIT_TOTAL] != 0)

                 && ((io_limits->bps[BLOCK_IO_LIMIT_READ] != 0)

                 || (io_limits->bps[BLOCK_IO_LIMIT_WRITE] != 0));

    iops_flag = (io_limits->iops[BLOCK_IO_LIMIT_TOTAL] != 0)

                 && ((io_limits->iops[BLOCK_IO_LIMIT_READ] != 0)

                 || (io_limits->iops[BLOCK_IO_LIMIT_WRITE] != 0));

    if (bps_flag || iops_flag) {

        return false;

    }



    return true;

}
