static int vtd_iova_to_slpte(VTDContextEntry *ce, uint64_t iova, bool is_write,

                             uint64_t *slptep, uint32_t *slpte_level,

                             bool *reads, bool *writes)

{

    dma_addr_t addr = vtd_get_slpt_base_from_context(ce);

    uint32_t level = vtd_get_level_from_context_entry(ce);

    uint32_t offset;

    uint64_t slpte;

    uint32_t ce_agaw = vtd_get_agaw_from_context_entry(ce);

    uint64_t access_right_check;



    /* Check if @iova is above 2^X-1, where X is the minimum of MGAW

     * in CAP_REG and AW in context-entry.

     */

    if (iova & ~((1ULL << MIN(ce_agaw, VTD_MGAW)) - 1)) {

        VTD_DPRINTF(GENERAL, "error: iova 0x%"PRIx64 " exceeds limits", iova);

        return -VTD_FR_ADDR_BEYOND_MGAW;

    }



    /* FIXME: what is the Atomics request here? */

    access_right_check = is_write ? VTD_SL_W : VTD_SL_R;



    while (true) {

        offset = vtd_iova_level_offset(iova, level);

        slpte = vtd_get_slpte(addr, offset);



        if (slpte == (uint64_t)-1) {

            VTD_DPRINTF(GENERAL, "error: fail to access second-level paging "

                        "entry at level %"PRIu32 " for iova 0x%"PRIx64,

                        level, iova);

            if (level == vtd_get_level_from_context_entry(ce)) {

                /* Invalid programming of context-entry */

                return -VTD_FR_CONTEXT_ENTRY_INV;

            } else {

                return -VTD_FR_PAGING_ENTRY_INV;

            }

        }

        *reads = (*reads) && (slpte & VTD_SL_R);

        *writes = (*writes) && (slpte & VTD_SL_W);

        if (!(slpte & access_right_check)) {

            VTD_DPRINTF(GENERAL, "error: lack of %s permission for "

                        "iova 0x%"PRIx64 " slpte 0x%"PRIx64,

                        (is_write ? "write" : "read"), iova, slpte);

            return is_write ? -VTD_FR_WRITE : -VTD_FR_READ;

        }

        if (vtd_slpte_nonzero_rsvd(slpte, level)) {

            VTD_DPRINTF(GENERAL, "error: non-zero reserved field in second "

                        "level paging entry level %"PRIu32 " slpte 0x%"PRIx64,

                        level, slpte);

            return -VTD_FR_PAGING_ENTRY_RSVD;

        }



        if (vtd_is_last_slpte(slpte, level)) {

            *slptep = slpte;

            *slpte_level = level;

            return 0;

        }

        addr = vtd_get_slpte_addr(slpte);

        level--;

    }

}
