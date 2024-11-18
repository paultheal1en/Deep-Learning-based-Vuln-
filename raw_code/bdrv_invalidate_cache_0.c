void bdrv_invalidate_cache(BlockDriverState *bs, Error **errp)

{

    Error *local_err = NULL;

    int ret;



    if (!bs->drv)  {

        return;

    }



    if (!(bs->open_flags & BDRV_O_INACTIVE)) {

        return;

    }

    bs->open_flags &= ~BDRV_O_INACTIVE;



    if (bs->drv->bdrv_invalidate_cache) {

        bs->drv->bdrv_invalidate_cache(bs, &local_err);

    } else if (bs->file) {

        bdrv_invalidate_cache(bs->file->bs, &local_err);

    }

    if (local_err) {

        bs->open_flags |= BDRV_O_INACTIVE;

        error_propagate(errp, local_err);

        return;

    }



    ret = refresh_total_sectors(bs, bs->total_sectors);

    if (ret < 0) {

        bs->open_flags |= BDRV_O_INACTIVE;

        error_setg_errno(errp, -ret, "Could not refresh total sector count");

        return;

    }

}
