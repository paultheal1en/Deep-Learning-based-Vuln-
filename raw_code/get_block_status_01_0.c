static int get_block_status(BlockDriverState *bs, int64_t offset,

                            int64_t bytes, MapEntry *e)

{

    int64_t ret;

    int depth;

    BlockDriverState *file;

    bool has_offset;

    int nb_sectors = bytes >> BDRV_SECTOR_BITS;



    assert(bytes < INT_MAX);

    /* As an optimization, we could cache the current range of unallocated

     * clusters in each file of the chain, and avoid querying the same

     * range repeatedly.

     */



    depth = 0;

    for (;;) {

        ret = bdrv_get_block_status(bs, offset >> BDRV_SECTOR_BITS, nb_sectors,

                                    &nb_sectors, &file);

        if (ret < 0) {

            return ret;

        }

        assert(nb_sectors);

        if (ret & (BDRV_BLOCK_ZERO|BDRV_BLOCK_DATA)) {

            break;

        }

        bs = backing_bs(bs);

        if (bs == NULL) {

            ret = 0;

            break;

        }



        depth++;

    }



    has_offset = !!(ret & BDRV_BLOCK_OFFSET_VALID);



    *e = (MapEntry) {

        .start = offset,

        .length = nb_sectors * BDRV_SECTOR_SIZE,

        .data = !!(ret & BDRV_BLOCK_DATA),

        .zero = !!(ret & BDRV_BLOCK_ZERO),

        .offset = ret & BDRV_BLOCK_OFFSET_MASK,

        .has_offset = has_offset,

        .depth = depth,

        .has_filename = file && has_offset,

        .filename = file && has_offset ? file->filename : NULL,

    };



    return 0;

}
