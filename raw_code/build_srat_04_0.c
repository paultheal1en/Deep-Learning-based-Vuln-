build_srat(GArray *table_data, BIOSLinker *linker, VirtMachineState *vms)

{

    AcpiSystemResourceAffinityTable *srat;

    AcpiSratProcessorGiccAffinity *core;

    AcpiSratMemoryAffinity *numamem;

    int i, srat_start;

    uint64_t mem_base;

    MachineClass *mc = MACHINE_GET_CLASS(vms);

    const CPUArchIdList *cpu_list = mc->possible_cpu_arch_ids(MACHINE(vms));



    srat_start = table_data->len;

    srat = acpi_data_push(table_data, sizeof(*srat));

    srat->reserved1 = cpu_to_le32(1);



    for (i = 0; i < cpu_list->len; ++i) {

        int node_id = cpu_list->cpus[i].props.has_node_id ?

            cpu_list->cpus[i].props.node_id : 0;

        core = acpi_data_push(table_data, sizeof(*core));

        core->type = ACPI_SRAT_PROCESSOR_GICC;

        core->length = sizeof(*core);

        core->proximity = cpu_to_le32(node_id);

        core->acpi_processor_uid = cpu_to_le32(i);

        core->flags = cpu_to_le32(1);

    }



    mem_base = vms->memmap[VIRT_MEM].base;

    for (i = 0; i < nb_numa_nodes; ++i) {

        numamem = acpi_data_push(table_data, sizeof(*numamem));

        build_srat_memory(numamem, mem_base, numa_info[i].node_mem, i,

                          MEM_AFFINITY_ENABLED);

        mem_base += numa_info[i].node_mem;

    }



    build_header(linker, table_data, (void *)srat, "SRAT",

                 table_data->len - srat_start, 3, NULL, NULL);

}
