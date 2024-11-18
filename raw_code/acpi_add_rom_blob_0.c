static MemoryRegion *acpi_add_rom_blob(AcpiBuildState *build_state,

                                       GArray *blob, const char *name,

                                       uint64_t max_size)

{

    return rom_add_blob(name, blob->data, acpi_data_len(blob), max_size, -1,

                        name, virt_acpi_build_update, build_state);

}
