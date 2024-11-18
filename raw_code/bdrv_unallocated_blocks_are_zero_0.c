bool bdrv_unallocated_blocks_are_zero(BlockDriverState *bs)

{

    BlockDriverInfo bdi;



    if (bs->backing_hd) {

        return false;

    }



    if (bdrv_get_info(bs, &bdi) == 0) {

        return bdi.unallocated_blocks_are_zero;

    }



    return false;

}
