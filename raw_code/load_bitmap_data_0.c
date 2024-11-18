static int load_bitmap_data(BlockDriverState *bs,

                            const uint64_t *bitmap_table,

                            uint32_t bitmap_table_size,

                            BdrvDirtyBitmap *bitmap)

{

    int ret = 0;

    BDRVQcow2State *s = bs->opaque;

    uint64_t sector, sbc;

    uint64_t bm_size = bdrv_dirty_bitmap_size(bitmap);

    uint64_t bm_sectors = DIV_ROUND_UP(bm_size, BDRV_SECTOR_SIZE);

    uint8_t *buf = NULL;

    uint64_t i, tab_size =

            size_to_clusters(s,

                bdrv_dirty_bitmap_serialization_size(bitmap, 0, bm_sectors));



    if (tab_size != bitmap_table_size || tab_size > BME_MAX_TABLE_SIZE) {

        return -EINVAL;

    }



    buf = g_malloc(s->cluster_size);

    sbc = sectors_covered_by_bitmap_cluster(s, bitmap);

    for (i = 0, sector = 0; i < tab_size; ++i, sector += sbc) {

        uint64_t count = MIN(bm_sectors - sector, sbc);

        uint64_t entry = bitmap_table[i];

        uint64_t offset = entry & BME_TABLE_ENTRY_OFFSET_MASK;



        assert(check_table_entry(entry, s->cluster_size) == 0);



        if (offset == 0) {

            if (entry & BME_TABLE_ENTRY_FLAG_ALL_ONES) {

                bdrv_dirty_bitmap_deserialize_ones(bitmap, sector, count,

                                                   false);

            } else {

                /* No need to deserialize zeros because the dirty bitmap is

                 * already cleared */

            }

        } else {

            ret = bdrv_pread(bs->file, offset, buf, s->cluster_size);

            if (ret < 0) {

                goto finish;

            }

            bdrv_dirty_bitmap_deserialize_part(bitmap, buf, sector, count,

                                               false);

        }

    }

    ret = 0;



    bdrv_dirty_bitmap_deserialize_finish(bitmap);



finish:

    g_free(buf);



    return ret;

}
