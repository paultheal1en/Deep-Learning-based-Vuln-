void xen_map_cache_init(void)

{

    unsigned long size;

    struct rlimit rlimit_as;



    mapcache = g_malloc0(sizeof (MapCache));



    QTAILQ_INIT(&mapcache->locked_entries);

    mapcache->last_address_index = -1;



    if (geteuid() == 0) {

        rlimit_as.rlim_cur = RLIM_INFINITY;

        rlimit_as.rlim_max = RLIM_INFINITY;

        mapcache->max_mcache_size = MCACHE_MAX_SIZE;

    } else {

        getrlimit(RLIMIT_AS, &rlimit_as);

        rlimit_as.rlim_cur = rlimit_as.rlim_max;



        if (rlimit_as.rlim_max != RLIM_INFINITY) {

            fprintf(stderr, "Warning: QEMU's maximum size of virtual"

                    " memory is not infinity.\n");

        }

        if (rlimit_as.rlim_max < MCACHE_MAX_SIZE + NON_MCACHE_MEMORY_SIZE) {

            mapcache->max_mcache_size = rlimit_as.rlim_max -

                NON_MCACHE_MEMORY_SIZE;

        } else {

            mapcache->max_mcache_size = MCACHE_MAX_SIZE;

        }

    }



    setrlimit(RLIMIT_AS, &rlimit_as);



    mapcache->nr_buckets =

        (((mapcache->max_mcache_size >> XC_PAGE_SHIFT) +

          (1UL << (MCACHE_BUCKET_SHIFT - XC_PAGE_SHIFT)) - 1) >>

         (MCACHE_BUCKET_SHIFT - XC_PAGE_SHIFT));



    size = mapcache->nr_buckets * sizeof (MapCacheEntry);

    size = (size + XC_PAGE_SIZE - 1) & ~(XC_PAGE_SIZE - 1);

    DPRINTF("%s, nr_buckets = %lx size %lu\n", __func__,

            mapcache->nr_buckets, size);

    mapcache->entry = g_malloc0(size);

}
