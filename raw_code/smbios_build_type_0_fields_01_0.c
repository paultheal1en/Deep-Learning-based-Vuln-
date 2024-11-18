static void smbios_build_type_0_fields(const char *t)

{

    char buf[1024];

    unsigned char major, minor;



    if (get_param_value(buf, sizeof(buf), "vendor", t))

        smbios_add_field(0, offsetof(struct smbios_type_0, vendor_str),

                         buf, strlen(buf) + 1);

    if (get_param_value(buf, sizeof(buf), "version", t))

        smbios_add_field(0, offsetof(struct smbios_type_0, bios_version_str),

                         buf, strlen(buf) + 1);

    if (get_param_value(buf, sizeof(buf), "date", t))

        smbios_add_field(0, offsetof(struct smbios_type_0,

                                     bios_release_date_str),

                         buf, strlen(buf) + 1);

    if (get_param_value(buf, sizeof(buf), "release", t)) {

        sscanf(buf, "%hhu.%hhu", &major, &minor);

        smbios_add_field(0, offsetof(struct smbios_type_0,

                                     system_bios_major_release),

                         &major, 1);

        smbios_add_field(0, offsetof(struct smbios_type_0,

                                     system_bios_minor_release),

                         &minor, 1);

    }

}
