static int bochs_read(BlockDriverState *bs, int64_t sector_num,

                    uint8_t *buf, int nb_sectors)

{

    int ret;



    while (nb_sectors > 0) {

        int64_t block_offset = seek_to_sector(bs, sector_num);

        if (block_offset >= 0) {

            ret = bdrv_pread(bs->file, block_offset, buf, 512);

            if (ret != 512) {

                return -1;

            }

        } else

            memset(buf, 0, 512);

        nb_sectors--;

        sector_num++;

        buf += 512;

    }

    return 0;

}
