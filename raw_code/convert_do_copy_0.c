static int convert_do_copy(ImgConvertState *s)

{

    uint8_t *buf = NULL;

    int64_t sector_num, allocated_done;

    int ret;

    int n;



    /* Check whether we have zero initialisation or can get it efficiently */

    s->has_zero_init = s->min_sparse && !s->target_has_backing

                     ? bdrv_has_zero_init(blk_bs(s->target))

                     : false;



    if (!s->has_zero_init && !s->target_has_backing &&

        bdrv_can_write_zeroes_with_unmap(blk_bs(s->target)))

    {

        ret = bdrv_make_zero(blk_bs(s->target), BDRV_REQ_MAY_UNMAP);

        if (ret == 0) {

            s->has_zero_init = true;

        }

    }



    /* Allocate buffer for copied data. For compressed images, only one cluster

     * can be copied at a time. */

    if (s->compressed) {

        if (s->cluster_sectors <= 0 || s->cluster_sectors > s->buf_sectors) {

            error_report("invalid cluster size");

            ret = -EINVAL;

            goto fail;

        }

        s->buf_sectors = s->cluster_sectors;

    }

    buf = blk_blockalign(s->target, s->buf_sectors * BDRV_SECTOR_SIZE);



    /* Calculate allocated sectors for progress */

    s->allocated_sectors = 0;

    sector_num = 0;

    while (sector_num < s->total_sectors) {

        n = convert_iteration_sectors(s, sector_num);

        if (n < 0) {

            ret = n;

            goto fail;

        }

        if (s->status == BLK_DATA) {

            s->allocated_sectors += n;

        }

        sector_num += n;

    }



    /* Do the copy */

    s->src_cur = 0;

    s->src_cur_offset = 0;

    s->sector_next_status = 0;



    sector_num = 0;

    allocated_done = 0;



    while (sector_num < s->total_sectors) {

        n = convert_iteration_sectors(s, sector_num);

        if (n < 0) {

            ret = n;

            goto fail;

        }

        if (s->status == BLK_DATA) {

            allocated_done += n;

            qemu_progress_print(100.0 * allocated_done / s->allocated_sectors,

                                0);

        }



        ret = convert_read(s, sector_num, n, buf);

        if (ret < 0) {

            error_report("error while reading sector %" PRId64

                         ": %s", sector_num, strerror(-ret));

            goto fail;

        }



        ret = convert_write(s, sector_num, n, buf);

        if (ret < 0) {

            error_report("error while writing sector %" PRId64

                         ": %s", sector_num, strerror(-ret));

            goto fail;

        }



        sector_num += n;

    }



    if (s->compressed) {

        /* signal EOF to align */

        ret = blk_write_compressed(s->target, 0, NULL, 0);

        if (ret < 0) {

            goto fail;

        }

    }



    ret = 0;

fail:

    qemu_vfree(buf);

    return ret;

}
