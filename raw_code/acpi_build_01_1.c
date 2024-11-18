void acpi_build(PcGuestInfo *guest_info, AcpiBuildTables *tables)

{

    GArray *table_offsets;

    unsigned facs, ssdt, dsdt, rsdt;

    AcpiCpuInfo cpu;

    AcpiPmInfo pm;

    AcpiMiscInfo misc;

    AcpiMcfgInfo mcfg;

    PcPciInfo pci;

    uint8_t *u;

    size_t aml_len = 0;



    acpi_get_cpu_info(&cpu);

    acpi_get_pm_info(&pm);

    acpi_get_dsdt(&misc);

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



    /* Count the size of the DSDT and SSDT, we will need it for legacy

     * sizing of ACPI tables.

     */

    aml_len += tables->table_data->len - dsdt;



    /* ACPI tables pointed to by RSDT */

    acpi_add_table(table_offsets, tables->table_data);

    build_fadt(tables->table_data, tables->linker, &pm, facs, dsdt);



    ssdt = tables->table_data->len;

    acpi_add_table(table_offsets, tables->table_data);

    build_ssdt(tables->table_data, tables->linker, &cpu, &pm, &misc, &pci,

               guest_info);

    aml_len += tables->table_data->len - ssdt;



    acpi_add_table(table_offsets, tables->table_data);

    build_madt(tables->table_data, tables->linker, &cpu, guest_info);



    if (misc.has_hpet) {

        acpi_add_table(table_offsets, tables->table_data);

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



    /* We'll expose it all to Guest so we want to reduce

     * chance of size changes.

     * RSDP is small so it's easy to keep it immutable, no need to

     * bother with alignment.

     *

     * We used to align the tables to 4k, but of course this would

     * too simple to be enough.  4k turned out to be too small an

     * alignment very soon, and in fact it is almost impossible to

     * keep the table size stable for all (max_cpus, max_memory_slots)

     * combinations.  So the table size is always 64k for pc-i440fx-2.1

     * and we give an error if the table grows beyond that limit.

     *

     * We still have the problem of migrating from "-M pc-i440fx-2.0".  For

     * that, we exploit the fact that QEMU 2.1 generates _smaller_ tables

     * than 2.0 and we can always pad the smaller tables with zeros.  We can

     * then use the exact size of the 2.0 tables.

     *

     * All this is for PIIX4, since QEMU 2.0 didn't support Q35 migration.

     */

    if (guest_info->legacy_acpi_table_size) {

        /* Subtracting aml_len gives the size of fixed tables.  Then add the

         * size of the PIIX4 DSDT/SSDT in QEMU 2.0.

         */

        int legacy_aml_len =

            guest_info->legacy_acpi_table_size +

            ACPI_BUILD_LEGACY_CPU_AML_SIZE * max_cpus;

        int legacy_table_size =

            ROUND_UP(tables->table_data->len - aml_len + legacy_aml_len,

                     ACPI_BUILD_ALIGN_SIZE);

        if (tables->table_data->len > legacy_table_size) {

            /* Should happen only with PCI bridges and -M pc-i440fx-2.0.  */

            error_report("Warning: migration to QEMU 2.0 may not work.");

        }

        g_array_set_size(tables->table_data, legacy_table_size);

    } else {

        if (tables->table_data->len > ACPI_BUILD_TABLE_SIZE) {

            /* As of QEMU 2.1, this fires with 160 VCPUs and 255 memory slots.  */

            error_report("ACPI tables are larger than 64k.  Please remove");

            error_report("CPUs, NUMA nodes, memory slots or PCI bridges.");

            exit(1);

        }

        g_array_set_size(tables->table_data, ACPI_BUILD_TABLE_SIZE);

    }



    acpi_align_size(tables->linker, ACPI_BUILD_ALIGN_SIZE);



    /* Cleanup memory that's no longer used. */

    g_array_free(table_offsets, true);

}
