static void spapr_add_lmbs(DeviceState *dev, uint64_t addr_start, uint64_t size,

                           uint32_t node, bool dedicated_hp_event_source,

                           Error **errp)

{

    sPAPRDRConnector *drc;

    uint32_t nr_lmbs = size/SPAPR_MEMORY_BLOCK_SIZE;

    int i, fdt_offset, fdt_size;

    void *fdt;

    uint64_t addr = addr_start;



    for (i = 0; i < nr_lmbs; i++) {

        drc = spapr_drc_by_id(TYPE_SPAPR_DRC_LMB,

                              addr / SPAPR_MEMORY_BLOCK_SIZE);

        g_assert(drc);



        fdt = create_device_tree(&fdt_size);

        fdt_offset = spapr_populate_memory_node(fdt, node, addr,

                                                SPAPR_MEMORY_BLOCK_SIZE);



        spapr_drc_attach(drc, dev, fdt, fdt_offset, !dev->hotplugged, errp);

        addr += SPAPR_MEMORY_BLOCK_SIZE;

        if (!dev->hotplugged) {

            sPAPRDRConnectorClass *drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);

            /* guests expect coldplugged LMBs to be pre-allocated */

            drck->set_allocation_state(drc, SPAPR_DR_ALLOCATION_STATE_USABLE);

            drck->set_isolation_state(drc, SPAPR_DR_ISOLATION_STATE_UNISOLATED);

        }

    }

    /* send hotplug notification to the

     * guest only in case of hotplugged memory

     */

    if (dev->hotplugged) {

        if (dedicated_hp_event_source) {

            drc = spapr_drc_by_id(TYPE_SPAPR_DRC_LMB,

                                  addr_start / SPAPR_MEMORY_BLOCK_SIZE);

            spapr_hotplug_req_add_by_count_indexed(SPAPR_DR_CONNECTOR_TYPE_LMB,

                                                   nr_lmbs,

                                                   spapr_drc_index(drc));

        } else {

            spapr_hotplug_req_add_by_count(SPAPR_DR_CONNECTOR_TYPE_LMB,

                                           nr_lmbs);

        }

    }

}
