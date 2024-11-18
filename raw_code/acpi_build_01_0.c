void acpi_build(PcGuestInfo *guest_info, AcpiBuildTables *tables)

{

    GArray *table_offsets;

    unsigned facs, dsdt, rsdt;

    AcpiCpuInfo cpu;

    AcpiPmInfo pm;

    AcpiMiscInfo misc;

    AcpiMcfgInfo mcfg;

    PcPciInfo pci;

    uint8_t *u;



    acpi_get_cpu_info(&cpu);

    acpi_get_pm_info(&pm);

    acpi_get_dsdt(&misc);

    acpi_get_hotplug_info(&misc);

    acpi_get_misc_info(&misc);

    acpi_get_pci_info(&pci);



    table_offsets = g_array_new(false, true /* clear */,

                                        sizeof(uint32_t));

    ACPI_BUILD_DPRINTF(3, "init ACPI tables\n");



    bios_linker_loader_alloc(tables->linker, ACPI_BUILD_TABLE_FILE,

                             64 /* Ensure FACS is aligned */,

                             false /* high memory */);



    /*

     * FACS is pointed to by FADT.

     * We place it first since it's the only table that has alignment

     * requirements.

     */

    facs = tables->table_data->len;

    build_facs(tables->table_data, tables->linker, guest_info);



    /* DSDT is pointed to by FADT */

    dsdt = tables->table_data->len;

    build_dsdt(tables->table_data, tables->linker, &misc);



    /* ACPI tables pointed to by RSDT */

    acpi_add_table(table_offsets, tables->table_data);

    build_fadt(tables->table_data, tables->linker, &pm, facs, dsdt);

    acpi_add_table(table_offsets, tables->table_data);



    build_ssdt(tables->table_data, tables->linker, &cpu, &pm, &misc, &pci,

               guest_info);

    acpi_add_table(table_offsets, tables->table_data);



    build_madt(tables->table_data, tables->linker, &cpu, guest_info);

    acpi_add_table(table_offsets, tables->table_data);

    if (misc.has_hpet) {

        build_hpet(tables->table_data, tables->linker);

    }

    if (guest_info->numa_nodes) {

        acpi_add_table(table_offsets, tables->table_data);

        build_srat(tables->table_data, tables->linker, &cpu, guest_info);

    }

    if (acpi_get_mcfg(&mcfg)) {

        acpi_add_table(table_offsets, tables->table_data);

        build_mcfg_q35(tables->table_data, tables->linker, &mcfg);

    }



    /* Add tables supplied by user (if any) */

    for (u = acpi_table_first(); u; u = acpi_table_next(u)) {

        unsigned len = acpi_table_len(u);



        acpi_add_table(table_offsets, tables->table_data);

        g_array_append_vals(tables->table_data, u, len);

    }



    /* RSDT is pointed to by RSDP */

    rsdt = tables->table_data->len;

    build_rsdt(tables->table_data, tables->linker, table_offsets);



    /* RSDP is in FSEG memory, so allocate it separately */

    build_rsdp(tables->rsdp, tables->linker, rsdt);



    /* We'll expose it all to Guest so align size to reduce

     * chance of size changes.

     * RSDP is small so it's easy to keep it immutable, no need to

     * bother with alignment.

     */

    acpi_align_size(tables->table_data, 0x1000);



    acpi_align_size(tables->linker, 0x1000);



    /* Cleanup memory that's no longer used. */

    g_array_free(table_offsets, true);

}
