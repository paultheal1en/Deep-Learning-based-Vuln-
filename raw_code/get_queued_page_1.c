static bool get_queued_page(RAMState *rs, PageSearchStatus *pss,

                            ram_addr_t *ram_addr_abs)

{

    RAMBlock  *block;

    ram_addr_t offset;

    bool dirty;



    do {

        block = unqueue_page(rs, &offset, ram_addr_abs);

        /*

         * We're sending this page, and since it's postcopy nothing else

         * will dirty it, and we must make sure it doesn't get sent again

         * even if this queue request was received after the background

         * search already sent it.

         */

        if (block) {

            unsigned long *bitmap;

            bitmap = atomic_rcu_read(&rs->ram_bitmap)->bmap;

            dirty = test_bit(*ram_addr_abs >> TARGET_PAGE_BITS, bitmap);

            if (!dirty) {

                trace_get_queued_page_not_dirty(

                    block->idstr, (uint64_t)offset,

                    (uint64_t)*ram_addr_abs,

                    test_bit(*ram_addr_abs >> TARGET_PAGE_BITS,

                         atomic_rcu_read(&rs->ram_bitmap)->unsentmap));

            } else {

                trace_get_queued_page(block->idstr,

                                      (uint64_t)offset,

                                      (uint64_t)*ram_addr_abs);

            }

        }



    } while (block && !dirty);



    if (block) {

        /*

         * As soon as we start servicing pages out of order, then we have

         * to kill the bulk stage, since the bulk stage assumes

         * in (migration_bitmap_find_and_reset_dirty) that every page is

         * dirty, that's no longer true.

         */

        rs->ram_bulk_stage = false;



        /*

         * We want the background search to continue from the queued page

         * since the guest is likely to want other pages near to the page

         * it just requested.

         */

        pss->block = block;

        pss->offset = offset;

    }



    return !!block;

}
