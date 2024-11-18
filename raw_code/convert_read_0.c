static int convert_read(ImgConvertState *s, int64_t sector_num, int nb_sectors,

                        uint8_t *buf)

{

    int n;

    int ret;



    if (s->status == BLK_ZERO || s->status == BLK_BACKING_FILE) {

        return 0;

    }



    assert(nb_sectors <= s->buf_sectors);

    while (nb_sectors > 0) {

        BlockBackend *blk;

        int64_t bs_sectors;



        /* In the case of compression with multiple source files, we can get a

         * nb_sectors that spreads into the next part. So we must be able to

         * read across multiple BDSes for one convert_read() call. */

        convert_select_part(s, sector_num);

        blk = s->src[s->src_cur];

        bs_sectors = s->src_sectors[s->src_cur];



        n = MIN(nb_sectors, bs_sectors - (sector_num - s->src_cur_offset));

        ret = blk_read(blk, sector_num - s->src_cur_offset, buf, n);

        if (ret < 0) {

            return ret;

        }



        sector_num += n;

        nb_sectors -= n;

        buf += n * BDRV_SECTOR_SIZE;

    }



    return 0;

}
