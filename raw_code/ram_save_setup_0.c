static int ram_save_setup(QEMUFile *f, void *opaque)

{

    RAMBlock *block;



    bytes_transferred = 0;

    reset_ram_globals();



    if (migrate_use_xbzrle()) {

        XBZRLE.cache = cache_init(migrate_xbzrle_cache_size() /

                                  TARGET_PAGE_SIZE,

                                  TARGET_PAGE_SIZE);

        if (!XBZRLE.cache) {

            DPRINTF("Error creating cache\n");

            return -1;

        }

        XBZRLE.encoded_buf = g_malloc0(TARGET_PAGE_SIZE);

        XBZRLE.current_buf = g_malloc(TARGET_PAGE_SIZE);

        acct_clear();

    }



    /* Make sure all dirty bits are set */

    QLIST_FOREACH(block, &ram_list.blocks, next) {

        migration_bitmap_set_dirty(block->mr, block->length);

    }



    memory_global_dirty_log_start();

    memory_global_sync_dirty_bitmap(get_system_memory());



    qemu_put_be64(f, ram_bytes_total() | RAM_SAVE_FLAG_MEM_SIZE);



    QLIST_FOREACH(block, &ram_list.blocks, next) {

        qemu_put_byte(f, strlen(block->idstr));

        qemu_put_buffer(f, (uint8_t *)block->idstr, strlen(block->idstr));

        qemu_put_be64(f, block->length);

    }



    qemu_put_be64(f, RAM_SAVE_FLAG_EOS);



    return 0;

}
