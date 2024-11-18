void spapr_tce_table_enable(sPAPRTCETable *tcet,

                            uint32_t page_shift, uint64_t bus_offset,

                            uint32_t nb_table)

{

    if (tcet->nb_table) {

        error_report("Warning: trying to enable already enabled TCE table");

        return;

    }



    tcet->bus_offset = bus_offset;

    tcet->page_shift = page_shift;

    tcet->nb_table = nb_table;

    tcet->table = spapr_tce_alloc_table(tcet->liobn,

                                        tcet->page_shift,

                                        tcet->bus_offset,

                                        tcet->nb_table,

                                        &tcet->fd,

                                        tcet->need_vfio);



    memory_region_set_size(&tcet->iommu,

                           (uint64_t)tcet->nb_table << tcet->page_shift);

    memory_region_add_subregion(&tcet->root, tcet->bus_offset, &tcet->iommu);

}
