static int vtd_page_walk_level(dma_addr_t addr, uint64_t start,

                               uint64_t end, vtd_page_walk_hook hook_fn,

                               void *private, uint32_t level,

                               bool read, bool write, bool notify_unmap)

{

    bool read_cur, write_cur, entry_valid;

    uint32_t offset;

    uint64_t slpte;

    uint64_t subpage_size, subpage_mask;

    IOMMUTLBEntry entry;

    uint64_t iova = start;

    uint64_t iova_next;

    int ret = 0;



    trace_vtd_page_walk_level(addr, level, start, end);



    subpage_size = 1ULL << vtd_slpt_level_shift(level);

    subpage_mask = vtd_slpt_level_page_mask(level);



    while (iova < end) {

        iova_next = (iova & subpage_mask) + subpage_size;



        offset = vtd_iova_level_offset(iova, level);

        slpte = vtd_get_slpte(addr, offset);



        if (slpte == (uint64_t)-1) {

            trace_vtd_page_walk_skip_read(iova, iova_next);

            goto next;

        }



        if (vtd_slpte_nonzero_rsvd(slpte, level)) {

            trace_vtd_page_walk_skip_reserve(iova, iova_next);

            goto next;

        }



        /* Permissions are stacked with parents' */

        read_cur = read && (slpte & VTD_SL_R);

        write_cur = write && (slpte & VTD_SL_W);



        /*

         * As long as we have either read/write permission, this is a

         * valid entry. The rule works for both page entries and page

         * table entries.

         */

        entry_valid = read_cur | write_cur;



        if (vtd_is_last_slpte(slpte, level)) {

            entry.target_as = &address_space_memory;

            entry.iova = iova & subpage_mask;

            /* NOTE: this is only meaningful if entry_valid == true */

            entry.translated_addr = vtd_get_slpte_addr(slpte);

            entry.addr_mask = ~subpage_mask;

            entry.perm = IOMMU_ACCESS_FLAG(read_cur, write_cur);

            if (!entry_valid && !notify_unmap) {

                trace_vtd_page_walk_skip_perm(iova, iova_next);

                goto next;

            }

            trace_vtd_page_walk_one(level, entry.iova, entry.translated_addr,

                                    entry.addr_mask, entry.perm);

            if (hook_fn) {

                ret = hook_fn(&entry, private);

                if (ret < 0) {

                    return ret;

                }

            }

        } else {

            if (!entry_valid) {

                trace_vtd_page_walk_skip_perm(iova, iova_next);

                goto next;

            }

            ret = vtd_page_walk_level(vtd_get_slpte_addr(slpte), iova,

                                      MIN(iova_next, end), hook_fn, private,

                                      level - 1, read_cur, write_cur,

                                      notify_unmap);

            if (ret < 0) {

                return ret;

            }

        }



next:

        iova = iova_next;

    }



    return 0;

}
