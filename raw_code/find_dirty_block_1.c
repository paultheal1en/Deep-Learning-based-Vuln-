static bool find_dirty_block(RAMState *rs, PageSearchStatus *pss,

                             bool *again, ram_addr_t *ram_addr_abs)

{

    pss->offset = migration_bitmap_find_dirty(rs, pss->block, pss->offset,

                                              ram_addr_abs);

    if (pss->complete_round && pss->block == rs->last_seen_block &&

        pss->offset >= rs->last_offset) {

        /*

         * We've been once around the RAM and haven't found anything.

         * Give up.

         */

        *again = false;

        return false;

    }

    if (pss->offset >= pss->block->used_length) {

        /* Didn't find anything in this RAM Block */

        pss->offset = 0;

        pss->block = QLIST_NEXT_RCU(pss->block, next);

        if (!pss->block) {

            /* Hit the end of the list */

            pss->block = QLIST_FIRST_RCU(&ram_list.blocks);

            /* Flag that we've looped */

            pss->complete_round = true;

            rs->ram_bulk_stage = false;

            if (migrate_use_xbzrle()) {

                /* If xbzrle is on, stop using the data compression at this

                 * point. In theory, xbzrle can do better than compression.

                 */

                flush_compressed_data(rs);

            }

        }

        /* Didn't find anything this time, but try again on the new block */

        *again = true;

        return false;

    } else {

        /* Can go around again, but... */

        *again = true;

        /* We've found something so probably don't need to */

        return true;

    }

}
