static void qcow2_cache_table_release(BlockDriverState *bs, Qcow2Cache *c,

                                      int i, int num_tables)

{

/* Using MADV_DONTNEED to discard memory is a Linux-specific feature */

#ifdef CONFIG_LINUX

    BDRVQcow2State *s = bs->opaque;

    void *t = qcow2_cache_get_table_addr(bs, c, i);

    int align = getpagesize();

    size_t mem_size = (size_t) s->cluster_size * num_tables;

    size_t offset = QEMU_ALIGN_UP((uintptr_t) t, align) - (uintptr_t) t;

    size_t length = QEMU_ALIGN_DOWN(mem_size - offset, align);

    if (length > 0) {

        madvise((uint8_t *) t + offset, length, MADV_DONTNEED);

    }

#endif

}
