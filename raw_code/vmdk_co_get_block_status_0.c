static int64_t coroutine_fn vmdk_co_get_block_status(BlockDriverState *bs,

        int64_t sector_num, int nb_sectors, int *pnum)

{

    BDRVVmdkState *s = bs->opaque;

    int64_t index_in_cluster, n, ret;

    uint64_t offset;

    VmdkExtent *extent;



    extent = find_extent(s, sector_num, NULL);

    if (!extent) {

        return 0;

    }

    qemu_co_mutex_lock(&s->lock);

    ret = get_cluster_offset(bs, extent, NULL,

                             sector_num * 512, false, &offset,

                             0, 0);

    qemu_co_mutex_unlock(&s->lock);



    switch (ret) {

    case VMDK_ERROR:

        ret = -EIO;

        break;

    case VMDK_UNALLOC:

        ret = 0;

        break;

    case VMDK_ZEROED:

        ret = BDRV_BLOCK_ZERO;

        break;

    case VMDK_OK:

        ret = BDRV_BLOCK_DATA;

        if (extent->file == bs->file && !extent->compressed) {

            ret |= BDRV_BLOCK_OFFSET_VALID | offset;

        }



        break;

    }



    index_in_cluster = vmdk_find_index_in_cluster(extent, sector_num);

    n = extent->cluster_sectors - index_in_cluster;

    if (n > nb_sectors) {

        n = nb_sectors;

    }

    *pnum = n;

    return ret;

}
