build_mcfg(GArray *table_data, GArray *linker, VirtGuestInfo *guest_info)

{

    AcpiTableMcfg *mcfg;

    const MemMapEntry *memmap = guest_info->memmap;

    int len = sizeof(*mcfg) + sizeof(mcfg->allocation[0]);



    mcfg = acpi_data_push(table_data, len);

    mcfg->allocation[0].address = cpu_to_le64(memmap[VIRT_PCIE_ECAM].base);



    /* Only a single allocation so no need to play with segments */

    mcfg->allocation[0].pci_segment = cpu_to_le16(0);

    mcfg->allocation[0].start_bus_number = 0;

    mcfg->allocation[0].end_bus_number = (memmap[VIRT_PCIE_ECAM].size

                                          / PCIE_MMCFG_SIZE_MIN) - 1;



    build_header(linker, table_data, (void *)mcfg, "MCFG", len, 1, NULL);

}
