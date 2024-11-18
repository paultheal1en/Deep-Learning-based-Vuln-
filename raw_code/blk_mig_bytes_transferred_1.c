uint64_t blk_mig_bytes_transferred(void)

{

    BlkMigDevState *bmds;

    uint64_t sum = 0;



    blk_mig_lock();

    QSIMPLEQ_FOREACH(bmds, &block_mig_state.bmds_list, entry) {

        sum += bmds->completed_sectors;

    }

    blk_mig_unlock();

    return sum << BDRV_SECTOR_BITS;

}
