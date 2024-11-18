sPAPRTCETable *spapr_tce_new_table(DeviceState *owner, uint32_t liobn, size_t window_size)

{

    sPAPRTCETable *tcet;



    if (spapr_tce_find_by_liobn(liobn)) {

        fprintf(stderr, "Attempted to create TCE table with duplicate"

                " LIOBN 0x%x\n", liobn);

        return NULL;

    }



    if (!window_size) {

        return NULL;

    }



    tcet = g_malloc0(sizeof(*tcet));

    tcet->liobn = liobn;

    tcet->window_size = window_size;



    if (kvm_enabled()) {

        tcet->table = kvmppc_create_spapr_tce(liobn,

                                              window_size,

                                              &tcet->fd);

    }



    if (!tcet->table) {

        size_t table_size = (window_size >> SPAPR_TCE_PAGE_SHIFT)

            * sizeof(sPAPRTCE);

        tcet->table = g_malloc0(table_size);

    }



#ifdef DEBUG_TCE

    fprintf(stderr, "spapr_iommu: New TCE table @ %p, liobn=0x%x, "

            "table @ %p, fd=%d\n", tcet, liobn, tcet->table, tcet->fd);

#endif



    memory_region_init_iommu(&tcet->iommu, OBJECT(owner), &spapr_iommu_ops,

                             "iommu-spapr", UINT64_MAX);



    QLIST_INSERT_HEAD(&spapr_tce_tables, tcet, list);



    return tcet;

}
