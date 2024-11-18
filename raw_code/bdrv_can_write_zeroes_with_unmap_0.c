bool bdrv_can_write_zeroes_with_unmap(BlockDriverState *bs)

{

    BlockDriverInfo bdi;



    if (bs->backing_hd || !(bs->open_flags & BDRV_O_UNMAP)) {

        return false;

    }



    if (bdrv_get_info(bs, &bdi) == 0) {

        return bdi.can_write_zeroes_with_unmap;

    }



    return false;

}
