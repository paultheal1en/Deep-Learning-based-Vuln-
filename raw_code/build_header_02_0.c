build_header(BIOSLinker *linker, GArray *table_data,

             AcpiTableHeader *h, const char *sig, int len, uint8_t rev,

             const char *oem_id, const char *oem_table_id)

{

    memcpy(&h->signature, sig, 4);

    h->length = cpu_to_le32(len);

    h->revision = rev;



    if (oem_id) {

        strncpy((char *)h->oem_id, oem_id, sizeof h->oem_id);

    } else {

        memcpy(h->oem_id, ACPI_BUILD_APPNAME6, 6);

    }



    if (oem_table_id) {

        strncpy((char *)h->oem_table_id, oem_table_id, sizeof(h->oem_table_id));

    } else {

        memcpy(h->oem_table_id, ACPI_BUILD_APPNAME4, 4);

        memcpy(h->oem_table_id + 4, sig, 4);

    }



    h->oem_revision = cpu_to_le32(1);

    memcpy(h->asl_compiler_id, ACPI_BUILD_APPNAME4, 4);

    h->asl_compiler_revision = cpu_to_le32(1);

    h->checksum = 0;

    /* Checksum to be filled in by Guest linker */

    bios_linker_loader_add_checksum(linker, ACPI_BUILD_TABLE_FILE,

                                    h, len, &h->checksum);

}
