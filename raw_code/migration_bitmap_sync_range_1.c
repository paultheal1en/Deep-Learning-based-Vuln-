static void migration_bitmap_sync_range(ram_addr_t start, ram_addr_t length)

{

    unsigned long *bitmap;

    bitmap = atomic_rcu_read(&migration_bitmap);

    migration_dirty_pages +=

        cpu_physical_memory_sync_dirty_bitmap(bitmap, start, length);

}
