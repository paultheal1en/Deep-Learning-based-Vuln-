int coroutine_fn bdrv_co_discard(BlockDriverState *bs, int64_t sector_num,

                                 int nb_sectors)

{

    int max_discard;



    if (!bs->drv) {

        return -ENOMEDIUM;

    } else if (bdrv_check_request(bs, sector_num, nb_sectors)) {

        return -EIO;

    } else if (bs->read_only) {

        return -EROFS;

    }



    bdrv_reset_dirty(bs, sector_num, nb_sectors);



    /* Do nothing if disabled.  */

    if (!(bs->open_flags & BDRV_O_UNMAP)) {

        return 0;

    }



    if (!bs->drv->bdrv_co_discard && !bs->drv->bdrv_aio_discard) {

        return 0;

    }



    max_discard = bs->bl.max_discard ?  bs->bl.max_discard : MAX_DISCARD_DEFAULT;

    while (nb_sectors > 0) {

        int ret;

        int num = nb_sectors;



        /* align request */

        if (bs->bl.discard_alignment &&

            num >= bs->bl.discard_alignment &&

            sector_num % bs->bl.discard_alignment) {

            if (num > bs->bl.discard_alignment) {

                num = bs->bl.discard_alignment;

            }

            num -= sector_num % bs->bl.discard_alignment;

        }



        /* limit request size */

        if (num > max_discard) {

            num = max_discard;

        }



        if (bs->drv->bdrv_co_discard) {

            ret = bs->drv->bdrv_co_discard(bs, sector_num, num);

        } else {

            BlockDriverAIOCB *acb;

            CoroutineIOCompletion co = {

                .coroutine = qemu_coroutine_self(),

            };



            acb = bs->drv->bdrv_aio_discard(bs, sector_num, nb_sectors,

                                            bdrv_co_io_em_complete, &co);

            if (acb == NULL) {

                return -EIO;

            } else {

                qemu_coroutine_yield();

                ret = co.ret;

            }

        }

        if (ret) {

            return ret;

        }



        sector_num += num;

        nb_sectors -= num;

    }

    return 0;

}
