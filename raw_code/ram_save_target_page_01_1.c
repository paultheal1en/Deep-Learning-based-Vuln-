static int ram_save_target_page(RAMState *rs, PageSearchStatus *pss,

                                bool last_stage, ram_addr_t dirty_ram_abs)

{

    int res = 0;



    /* Check the pages is dirty and if it is send it */

    if (migration_bitmap_clear_dirty(rs, dirty_ram_abs)) {

        unsigned long *unsentmap;

        /*

         * If xbzrle is on, stop using the data compression after first

         * round of migration even if compression is enabled. In theory,

         * xbzrle can do better than compression.

         */



        if (migrate_use_compression()

            && (rs->ram_bulk_stage || !migrate_use_xbzrle())) {

            res = ram_save_compressed_page(rs, pss, last_stage);

        } else {

            res = ram_save_page(rs, pss, last_stage);

        }



        if (res < 0) {

            return res;

        }

        unsentmap = atomic_rcu_read(&rs->ram_bitmap)->unsentmap;

        if (unsentmap) {

            clear_bit(dirty_ram_abs >> TARGET_PAGE_BITS, unsentmap);

        }

    }



    return res;

}
