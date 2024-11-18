static void vtd_update_iotlb(IntelIOMMUState *s, uint16_t source_id,

                             uint16_t domain_id, hwaddr addr, uint64_t slpte,

                             bool read_flags, bool write_flags)

{

    VTDIOTLBEntry *entry = g_malloc(sizeof(*entry));

    uint64_t *key = g_malloc(sizeof(*key));

    uint64_t gfn = addr >> VTD_PAGE_SHIFT_4K;



    VTD_DPRINTF(CACHE, "update iotlb sid 0x%"PRIx16 " gpa 0x%"PRIx64

                " slpte 0x%"PRIx64 " did 0x%"PRIx16, source_id, addr, slpte,

                domain_id);

    if (g_hash_table_size(s->iotlb) >= VTD_IOTLB_MAX_SIZE) {

        VTD_DPRINTF(CACHE, "iotlb exceeds size limit, forced to reset");

        vtd_reset_iotlb(s);

    }



    entry->gfn = gfn;

    entry->domain_id = domain_id;

    entry->slpte = slpte;

    entry->read_flags = read_flags;

    entry->write_flags = write_flags;

    *key = gfn | ((uint64_t)(source_id) << VTD_IOTLB_SID_SHIFT);

    g_hash_table_replace(s->iotlb, key, entry);

}
