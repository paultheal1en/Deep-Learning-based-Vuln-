static void spapr_memory_unplug_request(HotplugHandler *hotplug_dev,

                                        DeviceState *dev, Error **errp)

{

    sPAPRMachineState *spapr = SPAPR_MACHINE(hotplug_dev);

    Error *local_err = NULL;

    PCDIMMDevice *dimm = PC_DIMM(dev);

    PCDIMMDeviceClass *ddc = PC_DIMM_GET_CLASS(dimm);

    MemoryRegion *mr = ddc->get_memory_region(dimm);

    uint64_t size = memory_region_size(mr);

    uint32_t nr_lmbs = size / SPAPR_MEMORY_BLOCK_SIZE;

    uint64_t addr_start, addr;

    int i;

    sPAPRDRConnector *drc;

    addr_start = object_property_get_uint(OBJECT(dimm), PC_DIMM_ADDR_PROP,

                                         &local_err);

    if (local_err) {

        goto out;

    }



    spapr_pending_dimm_unplugs_add(spapr, nr_lmbs, dimm);



    addr = addr_start;

    for (i = 0; i < nr_lmbs; i++) {

        drc = spapr_drc_by_id(TYPE_SPAPR_DRC_LMB,

                              addr / SPAPR_MEMORY_BLOCK_SIZE);

        g_assert(drc);



        spapr_drc_detach(drc);

        addr += SPAPR_MEMORY_BLOCK_SIZE;

    }



    drc = spapr_drc_by_id(TYPE_SPAPR_DRC_LMB,

                          addr_start / SPAPR_MEMORY_BLOCK_SIZE);

    spapr_hotplug_req_remove_by_count_indexed(SPAPR_DR_CONNECTOR_TYPE_LMB,

                                              nr_lmbs, spapr_drc_index(drc));

out:

    error_propagate(errp, local_err);

}
