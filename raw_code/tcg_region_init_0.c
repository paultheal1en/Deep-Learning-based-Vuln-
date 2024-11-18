void tcg_region_init(void)

{

    void *buf = tcg_init_ctx.code_gen_buffer;

    void *aligned;

    size_t size = tcg_init_ctx.code_gen_buffer_size;

    size_t page_size = qemu_real_host_page_size;

    size_t region_size;

    size_t n_regions;

    size_t i;



    /* We do not yet support multiple TCG contexts, so use one region for now */

    n_regions = 1;



    /* The first region will be 'aligned - buf' bytes larger than the others */

    aligned = QEMU_ALIGN_PTR_UP(buf, page_size);

    g_assert(aligned < tcg_init_ctx.code_gen_buffer + size);

    /*

     * Make region_size a multiple of page_size, using aligned as the start.

     * As a result of this we might end up with a few extra pages at the end of

     * the buffer; we will assign those to the last region.

     */

    region_size = (size - (aligned - buf)) / n_regions;

    region_size = QEMU_ALIGN_DOWN(region_size, page_size);



    /* A region must have at least 2 pages; one code, one guard */

    g_assert(region_size >= 2 * page_size);



    /* init the region struct */

    qemu_mutex_init(&region.lock);

    region.n = n_regions;

    region.size = region_size - page_size;

    region.stride = region_size;

    region.start = buf;

    region.start_aligned = aligned;

    /* page-align the end, since its last page will be a guard page */

    region.end = QEMU_ALIGN_PTR_DOWN(buf + size, page_size);

    /* account for that last guard page */

    region.end -= page_size;



    /* set guard pages */

    for (i = 0; i < region.n; i++) {

        void *start, *end;

        int rc;



        tcg_region_bounds(i, &start, &end);

        rc = qemu_mprotect_none(end, page_size);

        g_assert(!rc);

    }



    /* We do not yet support multiple TCG contexts so allocate the region now */

    {

        bool err = tcg_region_initial_alloc__locked(tcg_ctx);



        g_assert(!err);

    }

}
