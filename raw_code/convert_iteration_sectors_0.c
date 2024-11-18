static int convert_iteration_sectors(ImgConvertState *s, int64_t sector_num)

{

    int64_t ret;

    int n;



    convert_select_part(s, sector_num);



    assert(s->total_sectors > sector_num);

    n = MIN(s->total_sectors - sector_num, BDRV_REQUEST_MAX_SECTORS);



    if (s->sector_next_status <= sector_num) {

        BlockDriverState *file;

        ret = bdrv_get_block_status(blk_bs(s->src[s->src_cur]),

                                    sector_num - s->src_cur_offset,

                                    n, &n, &file);

        if (ret < 0) {

            return ret;

        }



        if (ret & BDRV_BLOCK_ZERO) {

            s->status = BLK_ZERO;

        } else if (ret & BDRV_BLOCK_DATA) {

            s->status = BLK_DATA;

        } else if (!s->target_has_backing) {

            /* Without a target backing file we must copy over the contents of

             * the backing file as well. */

            /* TODO Check block status of the backing file chain to avoid

             * needlessly reading zeroes and limiting the iteration to the

             * buffer size */

            s->status = BLK_DATA;

        } else {

            s->status = BLK_BACKING_FILE;

        }



        s->sector_next_status = sector_num + n;

    }



    n = MIN(n, s->sector_next_status - sector_num);

    if (s->status == BLK_DATA) {

        n = MIN(n, s->buf_sectors);

    }



    /* We need to write complete clusters for compressed images, so if an

     * unallocated area is shorter than that, we must consider the whole

     * cluster allocated. */

    if (s->compressed) {

        if (n < s->cluster_sectors) {

            n = MIN(s->cluster_sectors, s->total_sectors - sector_num);

            s->status = BLK_DATA;

        } else {

            n = QEMU_ALIGN_DOWN(n, s->cluster_sectors);

        }

    }



    return n;

}
