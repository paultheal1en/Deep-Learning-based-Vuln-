void pc_dimm_memory_unplug(DeviceState *dev, MemoryHotplugState *hpms,

                           MemoryRegion *mr)

{

    PCDIMMDevice *dimm = PC_DIMM(dev);



    numa_unset_mem_node_id(dimm->addr, memory_region_size(mr), dimm->node);

    memory_region_del_subregion(&hpms->mr, mr);

    vmstate_unregister_ram(mr, dev);

}
