static void pc_fw_add_pflash_drv(void)

{

    QemuOpts *opts;

    QEMUMachine *machine;

    char *filename;



    if (bios_name == NULL) {

        bios_name = BIOS_FILENAME;

    }

    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);



    opts = drive_add(IF_PFLASH, -1, filename, "readonly=on");



    g_free(filename);



    if (opts == NULL) {

      return;

    }



    machine = find_default_machine();

    if (machine == NULL) {

      return;

    }



    drive_init(opts, machine->use_scsi);

}
