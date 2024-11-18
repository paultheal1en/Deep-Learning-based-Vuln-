void acpi_setup(void)

{

    PCMachineState *pcms = PC_MACHINE(qdev_get_machine());

    PCMachineClass *pcmc = PC_MACHINE_GET_CLASS(pcms);

    AcpiBuildTables tables;

    AcpiBuildState *build_state;



    if (!pcms->fw_cfg) {

        ACPI_BUILD_DPRINTF("No fw cfg. Bailing out.\n");

        return;

    }



    if (!pcmc->has_acpi_build) {

        ACPI_BUILD_DPRINTF("ACPI build disabled. Bailing out.\n");

        return;

    }



    if (!acpi_enabled) {

        ACPI_BUILD_DPRINTF("ACPI disabled. Bailing out.\n");

        return;

    }



    build_state = g_malloc0(sizeof *build_state);



    acpi_set_pci_info();



    acpi_build_tables_init(&tables);

    acpi_build(&tables, MACHINE(pcms));



    /* Now expose it all to Guest */

    build_state->table_mr = acpi_add_rom_blob(build_state, tables.table_data,

                                               ACPI_BUILD_TABLE_FILE,

                                               ACPI_BUILD_TABLE_MAX_SIZE);

    assert(build_state->table_mr != NULL);



    build_state->linker_mr =

        acpi_add_rom_blob(build_state, tables.linker, "etc/table-loader", 0);



    fw_cfg_add_file(pcms->fw_cfg, ACPI_BUILD_TPMLOG_FILE,

                    tables.tcpalog->data, acpi_data_len(tables.tcpalog));



    if (!pcmc->rsdp_in_ram) {

        /*

         * Keep for compatibility with old machine types.

         * Though RSDP is small, its contents isn't immutable, so

         * we'll update it along with the rest of tables on guest access.

         */

        uint32_t rsdp_size = acpi_data_len(tables.rsdp);



        build_state->rsdp = g_memdup(tables.rsdp->data, rsdp_size);

        fw_cfg_add_file_callback(pcms->fw_cfg, ACPI_BUILD_RSDP_FILE,

                                 acpi_build_update, build_state,

                                 build_state->rsdp, rsdp_size);

        build_state->rsdp_mr = NULL;

    } else {

        build_state->rsdp = NULL;

        build_state->rsdp_mr = acpi_add_rom_blob(build_state, tables.rsdp,

                                                  ACPI_BUILD_RSDP_FILE, 0);

    }



    qemu_register_reset(acpi_build_reset, build_state);

    acpi_build_reset(build_state);

    vmstate_register(NULL, 0, &vmstate_acpi_build, build_state);



    /* Cleanup tables but don't free the memory: we track it

     * in build_state.

     */

    acpi_build_tables_cleanup(&tables, false);

}
