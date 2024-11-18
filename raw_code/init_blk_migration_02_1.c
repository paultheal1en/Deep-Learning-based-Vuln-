static void init_blk_migration(QEMUFile *f)

{

    BlockDriverState *bs;

    BlkMigDevState *bmds;

    int64_t sectors;



    block_mig_state.submitted = 0;

    block_mig_state.read_done = 0;

    block_mig_state.transferred = 0;

    block_mig_state.total_sector_sum = 0;

    block_mig_state.prev_progress = -1;

    block_mig_state.bulk_completed = 0;

    block_mig_state.zero_blocks = migrate_zero_blocks();



    for (bs = bdrv_next(NULL); bs; bs = bdrv_next(bs)) {

        if (bdrv_is_read_only(bs)) {

            continue;

        }



        sectors = bdrv_nb_sectors(bs);

        if (sectors <= 0) {

            return;

        }



        bmds = g_new0(BlkMigDevState, 1);

        bmds->bs = bs;

        bmds->bulk_completed = 0;

        bmds->total_sectors = sectors;

        bmds->completed_sectors = 0;

        bmds->shared_base = block_mig_state.shared_base;

        alloc_aio_bitmap(bmds);

        error_setg(&bmds->blocker, "block device is in use by migration");

        bdrv_op_block_all(bs, bmds->blocker);

        bdrv_ref(bs);



        block_mig_state.total_sector_sum += sectors;



        if (bmds->shared_base) {

            DPRINTF("Start migration for %s with shared base image\n",

                    bdrv_get_device_name(bs));

        } else {

            DPRINTF("Start full migration for %s\n", bdrv_get_device_name(bs));

        }



        QSIMPLEQ_INSERT_TAIL(&block_mig_state.bmds_list, bmds, entry);

    }

}
