void acpi_setup(PcGuestInfo *guest_info)

{

    AcpiBuildTables tables;

    AcpiBuildState *build_state;



    if (!guest_info->fw_cfg) {

        ACPI_BUILD_DPRINTF("No fw cfg. Bailing out.\n");

        return;

    }



    if (!guest_info->has_acpi_build) {

        ACPI_BUILD_DPRINTF("ACPI build disabled. Bailing out.\n");

        return;

    }



    if (!acpi_enabled) {

        ACPI_BUILD_DPRINTF("ACPI disabled. Bailing out.\n");

        return;

    }



    build_state = g_malloc0(sizeof *build_state);



    build_state->guest_info = guest_info;



    acpi_set_pci_info();



    acpi_build_tables_init(&tables);

    acpi_build(build_state->guest_info, &tables);



    /* Now expose it all to Guest */

    build_state->table_ram = acpi_add_rom_blob(build_state, tables.table_data,

                                               ACPI_BUILD_TABLE_FILE,

                                               ACPI_BUILD_TABLE_MAX_SIZE);

    assert(build_state->table_ram != RAM_ADDR_MAX);

    build_state->table_size = acpi_data_len(tables.table_data);



    build_state->linker_ram =

        acpi_add_rom_blob(build_state, tables.linker, "etc/table-loader", 0);

    build_state->linker_size = acpi_data_len(tables.linker);



    fw_cfg_add_file(guest_info->fw_cfg, ACPI_BUILD_TPMLOG_FILE,

                    tables.tcpalog->data, acpi_data_len(tables.tcpalog));



    if (guest_info->has_immutable_rsdp) {

        /*

         * Keep for compatibility with old machine types.

         * Though RSDP is small, its contents isn't immutable, so

         * update it along with the rest of tables on guest access.

         */

        fw_cfg_add_file_callback(guest_info->fw_cfg, ACPI_BUILD_RSDP_FILE,

                                 acpi_build_update, build_state,

                                 tables.rsdp->data, acpi_data_len(tables.rsdp));

        build_state->rsdp = tables.rsdp->data;

    } else {

        build_state->rsdp = qemu_get_ram_ptr(

            acpi_add_rom_blob(build_state, tables.rsdp, ACPI_BUILD_RSDP_FILE, 0)

        );

    }



    qemu_register_reset(acpi_build_reset, build_state);

    acpi_build_reset(build_state);

    vmstate_register(NULL, 0, &vmstate_acpi_build, build_state);



    /* Cleanup tables but don't free the memory: we track it

     * in build_state.

     */

    acpi_build_tables_cleanup(&tables, false);

}
