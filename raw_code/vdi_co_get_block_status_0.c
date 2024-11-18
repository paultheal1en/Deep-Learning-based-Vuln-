static int64_t coroutine_fn vdi_co_get_block_status(BlockDriverState *bs,

        int64_t sector_num, int nb_sectors, int *pnum)

{

    /* TODO: Check for too large sector_num (in bdrv_is_allocated or here). */

    BDRVVdiState *s = (BDRVVdiState *)bs->opaque;

    size_t bmap_index = sector_num / s->block_sectors;

    size_t sector_in_block = sector_num % s->block_sectors;

    int n_sectors = s->block_sectors - sector_in_block;

    uint32_t bmap_entry = le32_to_cpu(s->bmap[bmap_index]);

    uint64_t offset;

    int result;



    logout("%p, %" PRId64 ", %d, %p\n", bs, sector_num, nb_sectors, pnum);

    if (n_sectors > nb_sectors) {

        n_sectors = nb_sectors;

    }

    *pnum = n_sectors;

    result = VDI_IS_ALLOCATED(bmap_entry);

    if (!result) {

        return 0;

    }



    offset = s->header.offset_data +

                              (uint64_t)bmap_entry * s->block_size +

                              sector_in_block * SECTOR_SIZE;

    return BDRV_BLOCK_DATA | BDRV_BLOCK_OFFSET_VALID | offset;

}
