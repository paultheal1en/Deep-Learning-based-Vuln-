static int bdrv_wr_badreq_sectors(BlockDriverState *bs,

                                  int64_t sector_num, int nb_sectors)

{

    if (sector_num < 0 ||

        nb_sectors < 0)

        return 1;



    if (sector_num > bs->total_sectors - nb_sectors) {

        if (bs->autogrow)

            bs->total_sectors = sector_num + nb_sectors;

        else

            return 1;

    }

    return 0;

}
