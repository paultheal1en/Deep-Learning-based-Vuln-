void bios_linker_loader_add_pointer(GArray *linker,

                                    const char *dest_file,

                                    const char *src_file,

                                    GArray *table, void *pointer,

                                    uint8_t pointer_size)

{

    BiosLinkerLoaderEntry entry;



    memset(&entry, 0, sizeof entry);

    strncpy(entry.pointer.dest_file, dest_file,

            sizeof entry.pointer.dest_file - 1);

    strncpy(entry.pointer.src_file, src_file,

            sizeof entry.pointer.src_file - 1);

    entry.command = cpu_to_le32(BIOS_LINKER_LOADER_COMMAND_ADD_POINTER);

    entry.pointer.offset = cpu_to_le32((gchar *)pointer - table->data);

    entry.pointer.size = pointer_size;

    assert(pointer_size == 1 || pointer_size == 2 ||

           pointer_size == 4 || pointer_size == 8);



    g_array_append_val(linker, entry);

}
