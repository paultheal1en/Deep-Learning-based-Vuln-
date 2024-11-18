ram_addr_t xen_ram_addr_from_mapcache(void *ptr)

{

    MapCacheEntry *entry = NULL;

    MapCacheRev *reventry;

    hwaddr paddr_index;

    hwaddr size;

    int found = 0;



    QTAILQ_FOREACH(reventry, &mapcache->locked_entries, next) {

        if (reventry->vaddr_req == ptr) {

            paddr_index = reventry->paddr_index;

            size = reventry->size;

            found = 1;

            break;

        }

    }

    if (!found) {

        fprintf(stderr, "%s, could not find %p\n", __func__, ptr);

        QTAILQ_FOREACH(reventry, &mapcache->locked_entries, next) {

            DPRINTF("   "TARGET_FMT_plx" -> %p is present\n", reventry->paddr_index,

                    reventry->vaddr_req);

        }

        abort();

        return 0;

    }



    entry = &mapcache->entry[paddr_index % mapcache->nr_buckets];

    while (entry && (entry->paddr_index != paddr_index || entry->size != size)) {

        entry = entry->next;

    }

    if (!entry) {

        DPRINTF("Trying to find address %p that is not in the mapcache!\n", ptr);

        return 0;

    }

    return (reventry->paddr_index << MCACHE_BUCKET_SHIFT) +

        ((unsigned long) ptr - (unsigned long) entry->vaddr_base);

}
