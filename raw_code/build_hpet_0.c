build_hpet(GArray *table_data, GArray *linker)

{

    Acpi20Hpet *hpet;



    hpet = acpi_data_push(table_data, sizeof(*hpet));

    /* Note timer_block_id value must be kept in sync with value advertised by

     * emulated hpet

     */

    hpet->timer_block_id = cpu_to_le32(0x8086a201);

    hpet->addr.address = cpu_to_le64(HPET_BASE);

    build_header(linker, table_data,

                 (void *)hpet, "HPET", sizeof(*hpet), 1, NULL);

}
