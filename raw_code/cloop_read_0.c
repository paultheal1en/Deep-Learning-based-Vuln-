static int cloop_read(BlockDriverState *bs, int64_t sector_num,

                    uint8_t *buf, int nb_sectors)

{

    BDRVCloopState *s = bs->opaque;

    int i;



    for (i = 0; i < nb_sectors; i++) {

        uint32_t sector_offset_in_block =

            ((sector_num + i) % s->sectors_per_block),

            block_num = (sector_num + i) / s->sectors_per_block;

        if (cloop_read_block(bs, block_num) != 0) {

            return -1;

        }

        memcpy(buf + i * 512,

            s->uncompressed_block + sector_offset_in_block * 512, 512);

    }

    return 0;

}
