build_mcfg_q35(GArray *table_data, GArray *linker, AcpiMcfgInfo *info)

{

    AcpiTableMcfg *mcfg;

    const char *sig;

    int len = sizeof(*mcfg) + 1 * sizeof(mcfg->allocation[0]);



    mcfg = acpi_data_push(table_data, len);

    mcfg->allocation[0].address = cpu_to_le64(info->mcfg_base);

    /* Only a single allocation so no need to play with segments */

    mcfg->allocation[0].pci_segment = cpu_to_le16(0);

    mcfg->allocation[0].start_bus_number = 0;

    mcfg->allocation[0].end_bus_number = PCIE_MMCFG_BUS(info->mcfg_size - 1);



    /* MCFG is used for ECAM which can be enabled or disabled by guest.

     * To avoid table size changes (which create migration issues),

     * always create the table even if there are no allocations,

     * but set the signature to a reserved value in this case.

     * ACPI spec requires OSPMs to ignore such tables.

     */

    if (info->mcfg_base == PCIE_BASE_ADDR_UNMAPPED) {

        /* Reserved signature: ignored by OSPM */

        sig = "QEMU";

    } else {

        sig = "MCFG";

    }

    build_header(linker, table_data, (void *)mcfg, sig, len, 1, NULL, NULL);

}
