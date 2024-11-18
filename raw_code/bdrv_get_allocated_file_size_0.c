int64_t bdrv_get_allocated_file_size(BlockDriverState *bs)

{

    BlockDriver *drv = bs->drv;

    if (!drv) {

        return -ENOMEDIUM;

    }

    if (drv->bdrv_get_allocated_file_size) {

        return drv->bdrv_get_allocated_file_size(bs);

    }

    if (bs->file) {

        return bdrv_get_allocated_file_size(bs->file);

    }

    return -ENOTSUP;

}
