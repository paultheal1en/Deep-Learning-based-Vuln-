ram_addr_t migration_bitmap_find_and_reset_dirty(RAMBlock *rb,

                                                 ram_addr_t start)

{

    unsigned long base = rb->offset >> TARGET_PAGE_BITS;

    unsigned long nr = base + (start >> TARGET_PAGE_BITS);

    uint64_t rb_size = rb->used_length;

    unsigned long size = base + (rb_size >> TARGET_PAGE_BITS);

    unsigned long *bitmap;



    unsigned long next;



    bitmap = atomic_rcu_read(&migration_bitmap);

    if (ram_bulk_stage && nr > base) {

        next = nr + 1;

    } else {

        next = find_next_bit(bitmap, size, nr);

    }



    if (next < size) {

        clear_bit(next, bitmap);

        migration_dirty_pages--;

    }

    return (next - base) << TARGET_PAGE_BITS;

}
