static void pc_machine_class_init(ObjectClass *oc, void *data)

{

    MachineClass *mc = MACHINE_CLASS(oc);

    PCMachineClass *pcmc = PC_MACHINE_CLASS(oc);

    HotplugHandlerClass *hc = HOTPLUG_HANDLER_CLASS(oc);

    NMIClass *nc = NMI_CLASS(oc);



    pcmc->get_hotplug_handler = mc->get_hotplug_handler;

    pcmc->pci_enabled = true;

    pcmc->has_acpi_build = true;

    pcmc->rsdp_in_ram = true;

    pcmc->smbios_defaults = true;

    pcmc->smbios_uuid_encoded = true;

    pcmc->gigabyte_align = true;

    pcmc->has_reserved_memory = true;

    pcmc->kvmclock_enabled = true;

    pcmc->enforce_aligned_dimm = true;

    /* BIOS ACPI tables: 128K. Other BIOS datastructures: less than 4K reported

     * to be used at the moment, 32K should be enough for a while.  */

    pcmc->acpi_data_size = 0x20000 + 0x8000;

    pcmc->save_tsc_khz = true;

    mc->get_hotplug_handler = pc_get_hotpug_handler;

    mc->cpu_index_to_socket_id = pc_cpu_index_to_socket_id;

    mc->possible_cpu_arch_ids = pc_possible_cpu_arch_ids;

    mc->query_hotpluggable_cpus = pc_query_hotpluggable_cpus;

    mc->default_boot_order = "cad";

    mc->hot_add_cpu = pc_hot_add_cpu;

    mc->max_cpus = 255;

    mc->reset = pc_machine_reset;

    hc->pre_plug = pc_machine_device_pre_plug_cb;

    hc->plug = pc_machine_device_plug_cb;

    hc->post_plug = pc_machine_device_post_plug_cb;

    hc->unplug_request = pc_machine_device_unplug_request_cb;

    hc->unplug = pc_machine_device_unplug_cb;

    nc->nmi_monitor_handler = x86_nmi;



    object_class_property_add(oc, PC_MACHINE_MEMHP_REGION_SIZE, "int",

        pc_machine_get_hotplug_memory_region_size, NULL,

        NULL, NULL, &error_abort);



    object_class_property_add(oc, PC_MACHINE_MAX_RAM_BELOW_4G, "size",

        pc_machine_get_max_ram_below_4g, pc_machine_set_max_ram_below_4g,

        NULL, NULL, &error_abort);



    object_class_property_set_description(oc, PC_MACHINE_MAX_RAM_BELOW_4G,

        "Maximum ram below the 4G boundary (32bit boundary)", &error_abort);



    object_class_property_add(oc, PC_MACHINE_SMM, "OnOffAuto",

        pc_machine_get_smm, pc_machine_set_smm,

        NULL, NULL, &error_abort);

    object_class_property_set_description(oc, PC_MACHINE_SMM,

        "Enable SMM (pc & q35)", &error_abort);



    object_class_property_add(oc, PC_MACHINE_VMPORT, "OnOffAuto",

        pc_machine_get_vmport, pc_machine_set_vmport,

        NULL, NULL, &error_abort);

    object_class_property_set_description(oc, PC_MACHINE_VMPORT,

        "Enable vmport (pc & q35)", &error_abort);



    object_class_property_add_bool(oc, PC_MACHINE_NVDIMM,

        pc_machine_get_nvdimm, pc_machine_set_nvdimm, &error_abort);

}
