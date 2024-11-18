static int64_t coroutine_fn qcow2_co_get_block_status(BlockDriverState *bs,

        int64_t sector_num, int nb_sectors, int *pnum)

{

    BDRVQcow2State *s = bs->opaque;

    uint64_t cluster_offset;

    int index_in_cluster, ret;

    int64_t status = 0;



    *pnum = nb_sectors;

    qemu_co_mutex_lock(&s->lock);

    ret = qcow2_get_cluster_offset(bs, sector_num << 9, pnum, &cluster_offset);

    qemu_co_mutex_unlock(&s->lock);

    if (ret < 0) {

        return ret;

    }



    if (cluster_offset != 0 && ret != QCOW2_CLUSTER_COMPRESSED &&

        !s->cipher) {

        index_in_cluster = sector_num & (s->cluster_sectors - 1);

        cluster_offset |= (index_in_cluster << BDRV_SECTOR_BITS);

        status |= BDRV_BLOCK_OFFSET_VALID | cluster_offset;

    }

    if (ret == QCOW2_CLUSTER_ZERO) {

        status |= BDRV_BLOCK_ZERO;

    } else if (ret != QCOW2_CLUSTER_UNALLOCATED) {

        status |= BDRV_BLOCK_DATA;

    }

    return status;

}
