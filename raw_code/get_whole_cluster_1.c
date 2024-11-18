static int get_whole_cluster(BlockDriverState *bs, uint64_t cluster_offset,

                             uint64_t offset, int allocate)

{

    uint64_t parent_cluster_offset;

    BDRVVmdkState *s = bs->opaque;

    uint8_t  whole_grain[s->cluster_sectors*512];        // 128 sectors * 512 bytes each = grain size 64KB



    // we will be here if it's first write on non-exist grain(cluster).

    // try to read from parent image, if exist

    if (s->hd->backing_hd) {

        BDRVVmdkState *ps = s->hd->backing_hd->opaque;



        if (!vmdk_is_cid_valid(bs))

            return -1;

        parent_cluster_offset = get_cluster_offset(s->hd->backing_hd, offset, allocate);

        if (bdrv_pread(ps->hd, parent_cluster_offset, whole_grain, ps->cluster_sectors*512) != 

                                                                            ps->cluster_sectors*512)

            return -1;



        if (bdrv_pwrite(s->hd, cluster_offset << 9, whole_grain, sizeof(whole_grain)) != 

                                                                            sizeof(whole_grain))

            return -1;

    }

    return 0;

}
