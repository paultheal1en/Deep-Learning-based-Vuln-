static int write_l2_entries(BlockDriverState *bs, uint64_t *l2_table,

    uint64_t l2_offset, int l2_index, int num)

{

    int l2_start_index = l2_index & ~(L1_ENTRIES_PER_SECTOR - 1);

    int start_offset = (8 * l2_index) & ~511;

    int end_offset = (8 * (l2_index + num) + 511) & ~511;

    size_t len = end_offset - start_offset;

    int ret;



    BLKDBG_EVENT(bs->file, BLKDBG_L2_UPDATE);

    ret = bdrv_pwrite(bs->file, l2_offset + start_offset,

        &l2_table[l2_start_index], len);

    if (ret < 0) {

        return ret;

    }



    return 0;

}
