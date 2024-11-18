void migration_bitmap_extend(ram_addr_t old, ram_addr_t new)

{

    /* called in qemu main thread, so there is

     * no writing race against this migration_bitmap

     */

    if (migration_bitmap) {

        unsigned long *old_bitmap = migration_bitmap, *bitmap;

        bitmap = bitmap_new(new);



        /* prevent migration_bitmap content from being set bit

         * by migration_bitmap_sync_range() at the same time.

         * it is safe to migration if migration_bitmap is cleared bit

         * at the same time.

         */

        qemu_mutex_lock(&migration_bitmap_mutex);

        bitmap_copy(bitmap, old_bitmap, old);

        bitmap_set(bitmap, old, new - old);

        atomic_rcu_set(&migration_bitmap, bitmap);

        qemu_mutex_unlock(&migration_bitmap_mutex);

        migration_dirty_pages += new - old;

        synchronize_rcu();

        g_free(old_bitmap);

    }

}
