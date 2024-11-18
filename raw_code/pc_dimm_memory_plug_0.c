void pc_dimm_memory_plug(DeviceState *dev, MemoryHotplugState *hpms,

                         MemoryRegion *mr, uint64_t align, Error **errp)

{

    int slot;

    MachineState *machine = MACHINE(qdev_get_machine());

    PCDIMMDevice *dimm = PC_DIMM(dev);

    Error *local_err = NULL;

    uint64_t existing_dimms_capacity = 0;

    uint64_t addr;



    addr = object_property_get_int(OBJECT(dimm), PC_DIMM_ADDR_PROP, &local_err);

    if (local_err) {

        goto out;

    }



    addr = pc_dimm_get_free_addr(hpms->base,

                                 memory_region_size(&hpms->mr),

                                 !addr ? NULL : &addr, align,

                                 memory_region_size(mr), &local_err);

    if (local_err) {

        goto out;

    }



    existing_dimms_capacity = pc_existing_dimms_capacity(&local_err);

    if (local_err) {

        goto out;

    }



    if (existing_dimms_capacity + memory_region_size(mr) >

        machine->maxram_size - machine->ram_size) {

        error_setg(&local_err, "not enough space, currently 0x%" PRIx64

                   " in use of total hot pluggable 0x" RAM_ADDR_FMT,

                   existing_dimms_capacity,

                   machine->maxram_size - machine->ram_size);

        goto out;

    }



    object_property_set_int(OBJECT(dev), addr, PC_DIMM_ADDR_PROP, &local_err);

    if (local_err) {

        goto out;

    }

    trace_mhp_pc_dimm_assigned_address(addr);



    slot = object_property_get_int(OBJECT(dev), PC_DIMM_SLOT_PROP, &local_err);

    if (local_err) {

        goto out;

    }



    slot = pc_dimm_get_free_slot(slot == PC_DIMM_UNASSIGNED_SLOT ? NULL : &slot,

                                 machine->ram_slots, &local_err);

    if (local_err) {

        goto out;

    }

    object_property_set_int(OBJECT(dev), slot, PC_DIMM_SLOT_PROP, &local_err);

    if (local_err) {

        goto out;

    }

    trace_mhp_pc_dimm_assigned_slot(slot);



    if (kvm_enabled() && !kvm_has_free_slot(machine)) {

        error_setg(&local_err, "hypervisor has no free memory slots left");

        goto out;

    }



    if (!vhost_has_free_slot()) {

        error_setg(&local_err, "a used vhost backend has no free"

                               " memory slots left");

        goto out;

    }



    memory_region_add_subregion(&hpms->mr, addr - hpms->base, mr);

    vmstate_register_ram(mr, dev);

    numa_set_mem_node_id(addr, memory_region_size(mr), dimm->node);



out:

    error_propagate(errp, local_err);

}
