static void xbzrle_cache_zero_page(ram_addr_t current_addr)

{

    if (ram_bulk_stage || !migrate_use_xbzrle()) {

        return;

    }



    /* We don't care if this fails to allocate a new cache page

     * as long as it updated an old one */

    cache_insert(XBZRLE.cache, current_addr, ZERO_TARGET_PAGE);

}
