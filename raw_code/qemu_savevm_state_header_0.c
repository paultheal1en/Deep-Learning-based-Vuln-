void qemu_savevm_state_header(QEMUFile *f)

{

    trace_savevm_state_header();

    qemu_put_be32(f, QEMU_VM_FILE_MAGIC);

    qemu_put_be32(f, QEMU_VM_FILE_VERSION);



    if (migrate_get_current()->send_configuration ||

        enforce_config_section()) {

        qemu_put_byte(f, QEMU_VM_CONFIGURATION);

        vmstate_save_state(f, &vmstate_configuration, &savevm_state, 0);

    }

}
