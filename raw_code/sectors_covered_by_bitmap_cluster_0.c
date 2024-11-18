static uint64_t sectors_covered_by_bitmap_cluster(const BDRVQcow2State *s,

                                                  const BdrvDirtyBitmap *bitmap)

{

    uint32_t sector_granularity =

            bdrv_dirty_bitmap_granularity(bitmap) >> BDRV_SECTOR_BITS;



    return (uint64_t)sector_granularity * (s->cluster_size << 3);

}
