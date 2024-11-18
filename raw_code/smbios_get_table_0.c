uint8_t *smbios_get_table(size_t *length)

{

    smbios_validate_table();

    *length = smbios_entries_len;

    return smbios_entries;

}
