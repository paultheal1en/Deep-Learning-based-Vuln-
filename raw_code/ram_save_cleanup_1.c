static void ram_save_cleanup(void *opaque)

{

    RAMState **rsp = opaque;

    RAMBlock *block;



    /* caller have hold iothread lock or is in a bh, so there is

     * no writing race against this migration_bitmap

     */

    memory_global_dirty_log_stop();



    QLIST_FOREACH_RCU(block, &ram_list.blocks, next) {

        g_free(block->bmap);

        block->bmap = NULL;

        g_free(block->unsentmap);

        block->unsentmap = NULL;

    }



    XBZRLE_cache_lock();

    if (XBZRLE.cache) {

        cache_fini(XBZRLE.cache);

        g_free(XBZRLE.encoded_buf);

        g_free(XBZRLE.current_buf);

        g_free(XBZRLE.zero_target_page);

        XBZRLE.cache = NULL;

        XBZRLE.encoded_buf = NULL;

        XBZRLE.current_buf = NULL;

        XBZRLE.zero_target_page = NULL;

    }

    XBZRLE_cache_unlock();

    compress_threads_save_cleanup();

    ram_state_cleanup(rsp);

}
