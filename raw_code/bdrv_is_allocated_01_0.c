int coroutine_fn bdrv_is_allocated(BlockDriverState *bs, int64_t offset,

                                   int64_t bytes, int64_t *pnum)

{

    BlockDriverState *file;

    int64_t sector_num = offset >> BDRV_SECTOR_BITS;

    int nb_sectors = bytes >> BDRV_SECTOR_BITS;

    int64_t ret;

    int psectors;



    assert(QEMU_IS_ALIGNED(offset, BDRV_SECTOR_SIZE));

    assert(QEMU_IS_ALIGNED(bytes, BDRV_SECTOR_SIZE) && bytes < INT_MAX);

    ret = bdrv_get_block_status(bs, sector_num, nb_sectors, &psectors,

                                &file);

    if (ret < 0) {

        return ret;

    }

    if (pnum) {

        *pnum = psectors * BDRV_SECTOR_SIZE;

    }

    return !!(ret & BDRV_BLOCK_ALLOCATED);

}
