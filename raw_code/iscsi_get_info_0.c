static int iscsi_get_info(BlockDriverState *bs, BlockDriverInfo *bdi)

{

    IscsiLun *iscsilun = bs->opaque;

    bdi->unallocated_blocks_are_zero = !!iscsilun->lbprz;

    bdi->can_write_zeroes_with_unmap = iscsilun->lbprz && iscsilun->lbp.lbpws;

    /* Guess the internal cluster (page) size of the iscsi target by the means

     * of opt_unmap_gran. Transfer the unmap granularity only if it has a

     * reasonable size for bdi->cluster_size */

    if (iscsilun->bl.opt_unmap_gran * iscsilun->block_size >= 64 * 1024 &&

        iscsilun->bl.opt_unmap_gran * iscsilun->block_size <= 16 * 1024 * 1024) {

        bdi->cluster_size = iscsilun->bl.opt_unmap_gran * iscsilun->block_size;

    }

    return 0;

}
