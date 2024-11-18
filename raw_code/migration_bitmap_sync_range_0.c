static void migration_bitmap_sync_range(ram_addr_t start, ram_addr_t length)

{

    migration_dirty_pages +=

        cpu_physical_memory_sync_dirty_bitmap(migration_bitmap, start, length);

}
