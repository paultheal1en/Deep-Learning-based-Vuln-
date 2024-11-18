static int ram_find_and_save_block(RAMState *rs, bool last_stage)

{

    PageSearchStatus pss;

    int pages = 0;

    bool again, found;

    ram_addr_t dirty_ram_abs; /* Address of the start of the dirty page in

                                 ram_addr_t space */



    /* No dirty page as there is zero RAM */

    if (!ram_bytes_total()) {

        return pages;

    }



    pss.block = rs->last_seen_block;

    pss.offset = rs->last_offset;

    pss.complete_round = false;



    if (!pss.block) {

        pss.block = QLIST_FIRST_RCU(&ram_list.blocks);

    }



    do {

        again = true;

        found = get_queued_page(rs, &pss, &dirty_ram_abs);



        if (!found) {

            /* priority queue empty, so just search for something dirty */

            found = find_dirty_block(rs, &pss, &again, &dirty_ram_abs);

        }



        if (found) {

            pages = ram_save_host_page(rs, &pss, last_stage, dirty_ram_abs);

        }

    } while (!pages && again);



    rs->last_seen_block = pss.block;

    rs->last_offset = pss.offset;



    return pages;

}
