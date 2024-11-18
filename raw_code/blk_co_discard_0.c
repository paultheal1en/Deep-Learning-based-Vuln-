int blk_co_discard(BlockBackend *blk, int64_t sector_num, int nb_sectors)

{

    int ret = blk_check_request(blk, sector_num, nb_sectors);

    if (ret < 0) {

        return ret;

    }



    return bdrv_co_discard(blk_bs(blk), sector_num, nb_sectors);

}
