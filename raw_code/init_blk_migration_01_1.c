static int init_blk_migration(QEMUFile *f)

{

    BlockDriverState *bs;

    BlkMigDevState *bmds;

    int64_t sectors;

    BdrvNextIterator it;

    int i, num_bs = 0;

    struct {

        BlkMigDevState *bmds;

        BlockDriverState *bs;

    } *bmds_bs;

    Error *local_err = NULL;

    int ret;



    block_mig_state.submitted = 0;

    block_mig_state.read_done = 0;

    block_mig_state.transferred = 0;

    block_mig_state.total_sector_sum = 0;

    block_mig_state.prev_progress = -1;

    block_mig_state.bulk_completed = 0;

    block_mig_state.zero_blocks = migrate_zero_blocks();



    for (bs = bdrv_first(&it); bs; bs = bdrv_next(&it)) {

        num_bs++;

    }

    bmds_bs = g_malloc0(num_bs * sizeof(*bmds_bs));



    for (i = 0, bs = bdrv_first(&it); bs; bs = bdrv_next(&it), i++) {

        if (bdrv_is_read_only(bs)) {

            continue;

        }



        sectors = bdrv_nb_sectors(bs);

        if (sectors <= 0) {

            ret = sectors;


            goto out;

        }



        bmds = g_new0(BlkMigDevState, 1);

        bmds->blk = blk_new(BLK_PERM_CONSISTENT_READ, BLK_PERM_ALL);

        bmds->blk_name = g_strdup(bdrv_get_device_name(bs));

        bmds->bulk_completed = 0;

        bmds->total_sectors = sectors;

        bmds->completed_sectors = 0;

        bmds->shared_base = migrate_use_block_incremental();



        assert(i < num_bs);

        bmds_bs[i].bmds = bmds;

        bmds_bs[i].bs = bs;



        block_mig_state.total_sector_sum += sectors;



        if (bmds->shared_base) {

            DPRINTF("Start migration for %s with shared base image\n",

                    bdrv_get_device_name(bs));

        } else {

            DPRINTF("Start full migration for %s\n", bdrv_get_device_name(bs));

        }



        QSIMPLEQ_INSERT_TAIL(&block_mig_state.bmds_list, bmds, entry);

    }



    /* Can only insert new BDSes now because doing so while iterating block

     * devices may end up in a deadlock (iterating the new BDSes, too). */

    for (i = 0; i < num_bs; i++) {

        BlkMigDevState *bmds = bmds_bs[i].bmds;

        BlockDriverState *bs = bmds_bs[i].bs;



        if (bmds) {

            ret = blk_insert_bs(bmds->blk, bs, &local_err);

            if (ret < 0) {

                error_report_err(local_err);

                goto out;

            }



            alloc_aio_bitmap(bmds);

            error_setg(&bmds->blocker, "block device is in use by migration");

            bdrv_op_block_all(bs, bmds->blocker);

        }

    }



    ret = 0;

out:

    g_free(bmds_bs);

    return ret;

}