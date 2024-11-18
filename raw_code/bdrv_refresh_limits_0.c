void bdrv_refresh_limits(BlockDriverState *bs, Error **errp)

{

    BlockDriver *drv = bs->drv;

    Error *local_err = NULL;



    memset(&bs->bl, 0, sizeof(bs->bl));



    if (!drv) {

        return;

    }



    /* Default alignment based on whether driver has byte interface */

    bs->request_alignment = drv->bdrv_co_preadv ? 1 : 512;



    /* Take some limits from the children as a default */

    if (bs->file) {

        bdrv_refresh_limits(bs->file->bs, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            return;

        }

        bs->bl.opt_transfer = bs->file->bs->bl.opt_transfer;

        bs->bl.max_transfer = bs->file->bs->bl.max_transfer;

        bs->bl.min_mem_alignment = bs->file->bs->bl.min_mem_alignment;

        bs->bl.opt_mem_alignment = bs->file->bs->bl.opt_mem_alignment;

        bs->bl.max_iov = bs->file->bs->bl.max_iov;

    } else {

        bs->bl.min_mem_alignment = 512;

        bs->bl.opt_mem_alignment = getpagesize();



        /* Safe default since most protocols use readv()/writev()/etc */

        bs->bl.max_iov = IOV_MAX;

    }



    if (bs->backing) {

        bdrv_refresh_limits(bs->backing->bs, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            return;

        }

        bs->bl.opt_transfer = MAX(bs->bl.opt_transfer,

                                  bs->backing->bs->bl.opt_transfer);

        bs->bl.max_transfer = MIN_NON_ZERO(bs->bl.max_transfer,

                                           bs->backing->bs->bl.max_transfer);

        bs->bl.opt_mem_alignment =

            MAX(bs->bl.opt_mem_alignment,

                bs->backing->bs->bl.opt_mem_alignment);

        bs->bl.min_mem_alignment =

            MAX(bs->bl.min_mem_alignment,

                bs->backing->bs->bl.min_mem_alignment);

        bs->bl.max_iov =

            MIN(bs->bl.max_iov,

                bs->backing->bs->bl.max_iov);

    }



    /* Then let the driver override it */

    if (drv->bdrv_refresh_limits) {

        drv->bdrv_refresh_limits(bs, errp);

    }

}
