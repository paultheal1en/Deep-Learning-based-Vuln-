void virt_acpi_setup(VirtGuestInfo *guest_info)

{

    AcpiBuildTables tables;

    AcpiBuildState *build_state;



    if (!guest_info->fw_cfg) {

        trace_virt_acpi_setup();

        return;

    }



    if (!acpi_enabled) {

        trace_virt_acpi_setup();

        return;

    }



    build_state = g_malloc0(sizeof *build_state);

    build_state->guest_info = guest_info;



    acpi_build_tables_init(&tables);

    virt_acpi_build(build_state->guest_info, &tables);



    /* Now expose it all to Guest */

    build_state->table_mr = acpi_add_rom_blob(build_state, tables.table_data,

                                               ACPI_BUILD_TABLE_FILE,

                                               ACPI_BUILD_TABLE_MAX_SIZE);

    assert(build_state->table_mr != NULL);



    build_state->linker_mr =

        acpi_add_rom_blob(build_state, tables.linker, "etc/table-loader", 0);



    fw_cfg_add_file(guest_info->fw_cfg, ACPI_BUILD_TPMLOG_FILE,

                    tables.tcpalog->data, acpi_data_len(tables.tcpalog));



    build_state->rsdp_mr = acpi_add_rom_blob(build_state, tables.rsdp,

                                              ACPI_BUILD_RSDP_FILE, 0);



    qemu_register_reset(virt_acpi_build_reset, build_state);

    virt_acpi_build_reset(build_state);

    vmstate_register(NULL, 0, &vmstate_virt_acpi_build, build_state);



    /* Cleanup tables but don't free the memory: we track it

     * in build_state.

     */

    acpi_build_tables_cleanup(&tables, false);

}
