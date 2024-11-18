void bios_linker_loader_add_checksum(BIOSLinker *linker, const char *file_name,

                                     void *start, unsigned size,

                                     uint8_t *checksum)

{

    BiosLinkerLoaderEntry entry;

    const BiosLinkerFileEntry *file = bios_linker_find_file(linker, file_name);

    ptrdiff_t checksum_offset = (gchar *)checksum - file->blob->data;

    ptrdiff_t start_offset = (gchar *)start - file->blob->data;



    assert(checksum_offset >= 0);

    assert(start_offset >= 0);

    assert(checksum_offset + 1 <= file->blob->len);

    assert(start_offset + size <= file->blob->len);

    assert(*checksum == 0x0);



    memset(&entry, 0, sizeof entry);

    strncpy(entry.cksum.file, file_name, sizeof entry.cksum.file - 1);

    entry.command = cpu_to_le32(BIOS_LINKER_LOADER_COMMAND_ADD_CHECKSUM);

    entry.cksum.offset = cpu_to_le32(checksum_offset);

    entry.cksum.start = cpu_to_le32(start_offset);

    entry.cksum.length = cpu_to_le32(size);



    g_array_append_vals(linker->cmd_blob, &entry, sizeof entry);

}
