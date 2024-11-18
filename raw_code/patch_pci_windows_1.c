static void patch_pci_windows(PcPciInfo *pci, uint8_t *start, unsigned size)

{

    *ACPI_BUILD_PTR(start, size, acpi_pci32_start[0], uint32_t) =

        cpu_to_le32(pci->w32.begin);



    *ACPI_BUILD_PTR(start, size, acpi_pci32_end[0], uint32_t) =

        cpu_to_le32(pci->w32.end - 1);



    if (pci->w64.end || pci->w64.begin) {

        *ACPI_BUILD_PTR(start, size, acpi_pci64_valid[0], uint8_t) = 1;

        *ACPI_BUILD_PTR(start, size, acpi_pci64_start[0], uint64_t) =

            cpu_to_le64(pci->w64.begin);

        *ACPI_BUILD_PTR(start, size, acpi_pci64_end[0], uint64_t) =

            cpu_to_le64(pci->w64.end - 1);

        *ACPI_BUILD_PTR(start, size, acpi_pci64_length[0], uint64_t) =

            cpu_to_le64(pci->w64.end - pci->w64.begin);

    } else {

        *ACPI_BUILD_PTR(start, size, acpi_pci64_valid[0], uint8_t) = 0;

    }

}
