static uint64_t *store_bitmap_data(BlockDriverState *bs,

                                   BdrvDirtyBitmap *bitmap,

                                   uint32_t *bitmap_table_size, Error **errp)

{

    int ret;

    BDRVQcow2State *s = bs->opaque;

    int64_t sector;

    uint64_t limit, sbc;

    uint64_t bm_size = bdrv_dirty_bitmap_size(bitmap);

    uint64_t bm_sectors = DIV_ROUND_UP(bm_size, BDRV_SECTOR_SIZE);

    const char *bm_name = bdrv_dirty_bitmap_name(bitmap);

    uint8_t *buf = NULL;

    BdrvDirtyBitmapIter *dbi;

    uint64_t *tb;

    uint64_t tb_size =

            size_to_clusters(s,

                bdrv_dirty_bitmap_serialization_size(bitmap, 0, bm_size));



    if (tb_size > BME_MAX_TABLE_SIZE ||

        tb_size * s->cluster_size > BME_MAX_PHYS_SIZE)

    {

        error_setg(errp, "Bitmap '%s' is too big", bm_name);

        return NULL;

    }



    tb = g_try_new0(uint64_t, tb_size);

    if (tb == NULL) {

        error_setg(errp, "No memory");

        return NULL;

    }



    dbi = bdrv_dirty_iter_new(bitmap);

    buf = g_malloc(s->cluster_size);

    limit = bytes_covered_by_bitmap_cluster(s, bitmap);

    sbc = limit >> BDRV_SECTOR_BITS;

    assert(DIV_ROUND_UP(bm_size, limit) == tb_size);



    while ((sector = bdrv_dirty_iter_next(dbi) >> BDRV_SECTOR_BITS) >= 0) {

        uint64_t cluster = sector / sbc;

        uint64_t end, write_size;

        int64_t off;



        sector = cluster * sbc;

        end = MIN(bm_sectors, sector + sbc);

        write_size = bdrv_dirty_bitmap_serialization_size(bitmap,

            sector * BDRV_SECTOR_SIZE, (end - sector) * BDRV_SECTOR_SIZE);

        assert(write_size <= s->cluster_size);



        off = qcow2_alloc_clusters(bs, s->cluster_size);

        if (off < 0) {

            error_setg_errno(errp, -off,

                             "Failed to allocate clusters for bitmap '%s'",

                             bm_name);

            goto fail;

        }

        tb[cluster] = off;



        bdrv_dirty_bitmap_serialize_part(bitmap, buf,

                                         sector * BDRV_SECTOR_SIZE,

                                         (end - sector) * BDRV_SECTOR_SIZE);

        if (write_size < s->cluster_size) {

            memset(buf + write_size, 0, s->cluster_size - write_size);

        }



        ret = qcow2_pre_write_overlap_check(bs, 0, off, s->cluster_size);

        if (ret < 0) {

            error_setg_errno(errp, -ret, "Qcow2 overlap check failed");

            goto fail;

        }



        ret = bdrv_pwrite(bs->file, off, buf, s->cluster_size);

        if (ret < 0) {

            error_setg_errno(errp, -ret, "Failed to write bitmap '%s' to file",

                             bm_name);

            goto fail;

        }



        if (end >= bm_sectors) {

            break;

        }



        bdrv_set_dirty_iter(dbi, end * BDRV_SECTOR_SIZE);

    }



    *bitmap_table_size = tb_size;

    g_free(buf);

    bdrv_dirty_iter_free(dbi);



    return tb;



fail:

    clear_bitmap_table(bs, tb, tb_size);

    g_free(buf);

    bdrv_dirty_iter_free(dbi);

    g_free(tb);



    return NULL;

}
