static void xen_remap_bucket(MapCacheEntry *entry,

                             hwaddr size,

                             hwaddr address_index)

{

    uint8_t *vaddr_base;

    xen_pfn_t *pfns;

    int *err;

    unsigned int i;

    hwaddr nb_pfn = size >> XC_PAGE_SHIFT;



    trace_xen_remap_bucket(address_index);



    pfns = g_malloc0(nb_pfn * sizeof (xen_pfn_t));

    err = g_malloc0(nb_pfn * sizeof (int));



    if (entry->vaddr_base != NULL) {

        if (munmap(entry->vaddr_base, entry->size) != 0) {

            perror("unmap fails");

            exit(-1);

        }

    }

    g_free(entry->valid_mapping);

    entry->valid_mapping = NULL;



    for (i = 0; i < nb_pfn; i++) {

        pfns[i] = (address_index << (MCACHE_BUCKET_SHIFT-XC_PAGE_SHIFT)) + i;

    }



    vaddr_base = xc_map_foreign_bulk(xen_xc, xen_domid, PROT_READ|PROT_WRITE,

                                     pfns, err, nb_pfn);

    if (vaddr_base == NULL) {

        perror("xc_map_foreign_bulk");

        exit(-1);

    }



    entry->vaddr_base = vaddr_base;

    entry->paddr_index = address_index;

    entry->size = size;

    entry->valid_mapping = (unsigned long *) g_malloc0(sizeof(unsigned long) *

            BITS_TO_LONGS(size >> XC_PAGE_SHIFT));



    bitmap_zero(entry->valid_mapping, nb_pfn);

    for (i = 0; i < nb_pfn; i++) {

        if (!err[i]) {

            bitmap_set(entry->valid_mapping, i, 1);

        }

    }



    g_free(pfns);

    g_free(err);

}
