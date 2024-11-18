static int vmdk_is_allocated(BlockDriverState *bs, int64_t sector_num,

                             int nb_sectors, int *pnum)

{

    BDRVVmdkState *s = bs->opaque;

    int64_t index_in_cluster, n, ret;

    uint64_t offset;

    VmdkExtent *extent;



    extent = find_extent(s, sector_num, NULL);

    if (!extent) {

        return 0;

    }

    ret = get_cluster_offset(bs, extent, NULL,

                            sector_num * 512, 0, &offset);

    /* get_cluster_offset returning 0 means success */

    ret = !ret;



    index_in_cluster = sector_num % extent->cluster_sectors;

    n = extent->cluster_sectors - index_in_cluster;

    if (n > nb_sectors) {

        n = nb_sectors;

    }

    *pnum = n;

    return ret;

}
