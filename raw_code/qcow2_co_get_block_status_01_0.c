static int64_t coroutine_fn qcow2_co_get_block_status(BlockDriverState *bs,

        int64_t sector_num, int nb_sectors, int *pnum, BlockDriverState **file)

{

    BDRVQcow2State *s = bs->opaque;

    uint64_t cluster_offset;

    int index_in_cluster, ret;

    unsigned int bytes;

    int64_t status = 0;



    bytes = MIN(INT_MAX, nb_sectors * BDRV_SECTOR_SIZE);

    qemu_co_mutex_lock(&s->lock);

    ret = qcow2_get_cluster_offset(bs, sector_num << 9, &bytes,

                                   &cluster_offset);

    qemu_co_mutex_unlock(&s->lock);

    if (ret < 0) {

        return ret;

    }



    *pnum = bytes >> BDRV_SECTOR_BITS;



    if (cluster_offset != 0 && ret != QCOW2_CLUSTER_COMPRESSED &&

        !s->cipher) {

        index_in_cluster = sector_num & (s->cluster_sectors - 1);

        cluster_offset |= (index_in_cluster << BDRV_SECTOR_BITS);

        *file = bs->file->bs;

        status |= BDRV_BLOCK_OFFSET_VALID | cluster_offset;

    }

    if (ret == QCOW2_CLUSTER_ZERO_PLAIN || ret == QCOW2_CLUSTER_ZERO_ALLOC) {

        status |= BDRV_BLOCK_ZERO;

    } else if (ret != QCOW2_CLUSTER_UNALLOCATED) {

        status |= BDRV_BLOCK_DATA;

    }

    return status;

}
