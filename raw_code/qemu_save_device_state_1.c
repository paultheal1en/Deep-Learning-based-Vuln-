static int qemu_save_device_state(QEMUFile *f)
{
    SaveStateEntry *se;
    qemu_put_be32(f, QEMU_VM_FILE_MAGIC);
    qemu_put_be32(f, QEMU_VM_FILE_VERSION);
    cpu_synchronize_all_states();
    QTAILQ_FOREACH(se, &savevm_state.handlers, entry) {
        if (se->is_ram) {
            continue;
        }
        if ((!se->ops || !se->ops->save_state) && !se->vmsd) {
            continue;
        }
        save_section_header(f, se, QEMU_VM_SECTION_FULL);
        vmstate_save(f, se, NULL);
    }
    qemu_put_byte(f, QEMU_VM_EOF);
    return qemu_file_get_error(f);
}