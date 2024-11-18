ram_addr_t migration_bitmap_find_dirty(RAMState *rs, RAMBlock *rb,

                                       ram_addr_t start,

                                       ram_addr_t *ram_addr_abs)

{

    unsigned long base = rb->offset >> TARGET_PAGE_BITS;

    unsigned long nr = base + (start >> TARGET_PAGE_BITS);

    uint64_t rb_size = rb->used_length;

    unsigned long size = base + (rb_size >> TARGET_PAGE_BITS);

    unsigned long *bitmap;



    unsigned long next;



    bitmap = atomic_rcu_read(&rs->ram_bitmap)->bmap;

    if (rs->ram_bulk_stage && nr > base) {

        next = nr + 1;

    } else {

        next = find_next_bit(bitmap, size, nr);

    }



    *ram_addr_abs = next << TARGET_PAGE_BITS;

    return (next - base) << TARGET_PAGE_BITS;

}
