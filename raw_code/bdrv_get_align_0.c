static inline uint64_t bdrv_get_align(BlockDriverState *bs)

{

    /* TODO Lift BDRV_SECTOR_SIZE restriction in BlockDriver interface */

    return MAX(BDRV_SECTOR_SIZE, bs->request_alignment);

}
