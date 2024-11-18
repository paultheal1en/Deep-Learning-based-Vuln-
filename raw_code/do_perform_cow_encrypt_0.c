static bool coroutine_fn do_perform_cow_encrypt(BlockDriverState *bs,

                                                uint64_t src_cluster_offset,

                                                unsigned offset_in_cluster,

                                                uint8_t *buffer,

                                                unsigned bytes)

{

    if (bytes && bs->encrypted) {

        BDRVQcow2State *s = bs->opaque;

        int64_t sector = (src_cluster_offset + offset_in_cluster)

                         >> BDRV_SECTOR_BITS;

        assert(s->cipher);

        assert((offset_in_cluster & ~BDRV_SECTOR_MASK) == 0);

        assert((bytes & ~BDRV_SECTOR_MASK) == 0);

        if (qcow2_encrypt_sectors(s, sector, buffer,

                                  bytes >> BDRV_SECTOR_BITS, true, NULL) < 0) {

            return false;

        }

    }

    return true;

}
