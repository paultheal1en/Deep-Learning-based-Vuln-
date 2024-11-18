static int64_t coroutine_fn parallels_co_get_block_status(BlockDriverState *bs,

        int64_t sector_num, int nb_sectors, int *pnum)

{

    BDRVParallelsState *s = bs->opaque;

    int64_t offset;



    qemu_co_mutex_lock(&s->lock);

    offset = block_status(s, sector_num, nb_sectors, pnum);

    qemu_co_mutex_unlock(&s->lock);



    if (offset < 0) {

        return 0;

    }



    return (offset << BDRV_SECTOR_BITS) |

        BDRV_BLOCK_DATA | BDRV_BLOCK_OFFSET_VALID;

}
