void virt_acpi_build(VirtMachineState *vms, AcpiBuildTables *tables)

{

    VirtMachineClass *vmc = VIRT_MACHINE_GET_CLASS(vms);

    GArray *table_offsets;

    unsigned dsdt, rsdt;

    GArray *tables_blob = tables->table_data;



    table_offsets = g_array_new(false, true /* clear */,

                                        sizeof(uint32_t));



    bios_linker_loader_alloc(tables->linker,

                             ACPI_BUILD_TABLE_FILE, tables_blob,

                             64, false /* high memory */);



    /* DSDT is pointed to by FADT */

    dsdt = tables_blob->len;

    build_dsdt(tables_blob, tables->linker, vms);



    /* FADT MADT GTDT MCFG SPCR pointed to by RSDT */

    acpi_add_table(table_offsets, tables_blob);

    build_fadt(tables_blob, tables->linker, vms, dsdt);



    acpi_add_table(table_offsets, tables_blob);

    build_madt(tables_blob, tables->linker, vms);



    acpi_add_table(table_offsets, tables_blob);

    build_gtdt(tables_blob, tables->linker, vms);



    acpi_add_table(table_offsets, tables_blob);

    build_mcfg(tables_blob, tables->linker, vms);



    acpi_add_table(table_offsets, tables_blob);

    build_spcr(tables_blob, tables->linker, vms);



    if (nb_numa_nodes > 0) {

        acpi_add_table(table_offsets, tables_blob);

        build_srat(tables_blob, tables->linker, vms);

    }



    if (its_class_name() && !vmc->no_its) {

        acpi_add_table(table_offsets, tables_blob);

        build_iort(tables_blob, tables->linker);

    }



    /* RSDT is pointed to by RSDP */

    rsdt = tables_blob->len;

    build_rsdt(tables_blob, tables->linker, table_offsets, NULL, NULL);



    /* RSDP is in FSEG memory, so allocate it separately */

    build_rsdp(tables->rsdp, tables->linker, rsdt);



    /* Cleanup memory that's no longer used. */

    g_array_free(table_offsets, true);

}
