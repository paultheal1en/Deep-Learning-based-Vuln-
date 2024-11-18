static int64_t coroutine_fn cow_co_get_block_status(BlockDriverState *bs,

        int64_t sector_num, int nb_sectors, int *num_same)

{

    BDRVCowState *s = bs->opaque;

    int ret = cow_co_is_allocated(bs, sector_num, nb_sectors, num_same);

    int64_t offset = s->cow_sectors_offset + (sector_num << BDRV_SECTOR_BITS);

    if (ret < 0) {

        return ret;

    }

    return (ret ? BDRV_BLOCK_DATA : 0) | offset | BDRV_BLOCK_OFFSET_VALID;

}
