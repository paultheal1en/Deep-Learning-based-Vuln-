static void pc_machine_initfn(Object *obj)

{

    PCMachineState *pcms = PC_MACHINE(obj);



    object_property_add(obj, PC_MACHINE_MEMHP_REGION_SIZE, "int",

                        pc_machine_get_hotplug_memory_region_size,

                        NULL, NULL, NULL, &error_abort);



    pcms->max_ram_below_4g = 0xe0000000; /* 3.5G */

    object_property_add(obj, PC_MACHINE_MAX_RAM_BELOW_4G, "size",

                        pc_machine_get_max_ram_below_4g,

                        pc_machine_set_max_ram_below_4g,

                        NULL, NULL, &error_abort);

    object_property_set_description(obj, PC_MACHINE_MAX_RAM_BELOW_4G,

                                    "Maximum ram below the 4G boundary (32bit boundary)",

                                    &error_abort);



    pcms->smm = ON_OFF_AUTO_AUTO;

    object_property_add(obj, PC_MACHINE_SMM, "OnOffAuto",

                        pc_machine_get_smm,

                        pc_machine_set_smm,

                        NULL, NULL, &error_abort);

    object_property_set_description(obj, PC_MACHINE_SMM,

                                    "Enable SMM (pc & q35)",

                                    &error_abort);



    pcms->vmport = ON_OFF_AUTO_AUTO;

    object_property_add(obj, PC_MACHINE_VMPORT, "OnOffAuto",

                        pc_machine_get_vmport,

                        pc_machine_set_vmport,

                        NULL, NULL, &error_abort);

    object_property_set_description(obj, PC_MACHINE_VMPORT,

                                    "Enable vmport (pc & q35)",

                                    &error_abort);



    /* nvdimm is disabled on default. */

    pcms->acpi_nvdimm_state.is_enabled = false;

    object_property_add_bool(obj, PC_MACHINE_NVDIMM, pc_machine_get_nvdimm,

                             pc_machine_set_nvdimm, &error_abort);

}
