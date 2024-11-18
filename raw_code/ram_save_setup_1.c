static int ram_save_setup(QEMUFile *f, void *opaque)

{

    RAMBlock *block;

    int64_t ram_pages = last_ram_offset() >> TARGET_PAGE_BITS;



    migration_bitmap = bitmap_new(ram_pages);

    bitmap_set(migration_bitmap, 0, ram_pages);

    migration_dirty_pages = ram_pages;

    mig_throttle_on = false;

    dirty_rate_high_cnt = 0;



    if (migrate_use_xbzrle()) {

        qemu_mutex_lock_iothread();

        XBZRLE.cache = cache_init(migrate_xbzrle_cache_size() /

                                  TARGET_PAGE_SIZE,

                                  TARGET_PAGE_SIZE);

        if (!XBZRLE.cache) {

            qemu_mutex_unlock_iothread();

            DPRINTF("Error creating cache\n");

            return -1;

        }

        qemu_mutex_init(&XBZRLE.lock);

        qemu_mutex_unlock_iothread();



        /* We prefer not to abort if there is no memory */

        XBZRLE.encoded_buf = g_try_malloc0(TARGET_PAGE_SIZE);

        if (!XBZRLE.encoded_buf) {

            DPRINTF("Error allocating encoded_buf\n");

            return -1;

        }



        XBZRLE.current_buf = g_try_malloc(TARGET_PAGE_SIZE);

        if (!XBZRLE.current_buf) {

            DPRINTF("Error allocating current_buf\n");

            g_free(XBZRLE.encoded_buf);

            XBZRLE.encoded_buf = NULL;

            return -1;

        }



        acct_clear();

    }



    qemu_mutex_lock_iothread();

    qemu_mutex_lock_ramlist();

    bytes_transferred = 0;

    reset_ram_globals();



    memory_global_dirty_log_start();

    migration_bitmap_sync();

    qemu_mutex_unlock_iothread();



    qemu_put_be64(f, ram_bytes_total() | RAM_SAVE_FLAG_MEM_SIZE);



    QTAILQ_FOREACH(block, &ram_list.blocks, next) {

        qemu_put_byte(f, strlen(block->idstr));

        qemu_put_buffer(f, (uint8_t *)block->idstr, strlen(block->idstr));

        qemu_put_be64(f, block->length);

    }



    qemu_mutex_unlock_ramlist();



    ram_control_before_iterate(f, RAM_CONTROL_SETUP);

    ram_control_after_iterate(f, RAM_CONTROL_SETUP);



    qemu_put_be64(f, RAM_SAVE_FLAG_EOS);



    return 0;

}
