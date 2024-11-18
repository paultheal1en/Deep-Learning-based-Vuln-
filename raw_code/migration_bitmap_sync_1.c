static void migration_bitmap_sync(void)

{

    RAMBlock *block;

    uint64_t num_dirty_pages_init = migration_dirty_pages;

    MigrationState *s = migrate_get_current();

    int64_t end_time;

    int64_t bytes_xfer_now;



    bitmap_sync_count++;



    if (!bytes_xfer_prev) {

        bytes_xfer_prev = ram_bytes_transferred();

    }



    if (!start_time) {

        start_time = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);

    }



    trace_migration_bitmap_sync_start();

    address_space_sync_dirty_bitmap(&address_space_memory);




    rcu_read_lock();

    QLIST_FOREACH_RCU(block, &ram_list.blocks, next) {

        migration_bitmap_sync_range(block->mr->ram_addr, block->used_length);

    }

    rcu_read_unlock();




    trace_migration_bitmap_sync_end(migration_dirty_pages

                                    - num_dirty_pages_init);

    num_dirty_pages_period += migration_dirty_pages - num_dirty_pages_init;

    end_time = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);



    /* more than 1 second = 1000 millisecons */

    if (end_time > start_time + 1000) {

        if (migrate_auto_converge()) {

            /* The following detection logic can be refined later. For now:

               Check to see if the dirtied bytes is 50% more than the approx.

               amount of bytes that just got transferred since the last time we

               were in this routine. If that happens >N times (for now N==4)

               we turn on the throttle down logic */

            bytes_xfer_now = ram_bytes_transferred();

            if (s->dirty_pages_rate &&

               (num_dirty_pages_period * TARGET_PAGE_SIZE >

                   (bytes_xfer_now - bytes_xfer_prev)/2) &&

               (dirty_rate_high_cnt++ > 4)) {

                    trace_migration_throttle();

                    mig_throttle_on = true;

                    dirty_rate_high_cnt = 0;

             }

             bytes_xfer_prev = bytes_xfer_now;

        } else {

             mig_throttle_on = false;

        }

        if (migrate_use_xbzrle()) {

            if (iterations_prev != acct_info.iterations) {

                acct_info.xbzrle_cache_miss_rate =

                   (double)(acct_info.xbzrle_cache_miss -

                            xbzrle_cache_miss_prev) /

                   (acct_info.iterations - iterations_prev);

            }

            iterations_prev = acct_info.iterations;

            xbzrle_cache_miss_prev = acct_info.xbzrle_cache_miss;

        }

        s->dirty_pages_rate = num_dirty_pages_period * 1000

            / (end_time - start_time);

        s->dirty_bytes_rate = s->dirty_pages_rate * TARGET_PAGE_SIZE;

        start_time = end_time;

        num_dirty_pages_period = 0;

    }

    s->dirty_sync_count = bitmap_sync_count;

}