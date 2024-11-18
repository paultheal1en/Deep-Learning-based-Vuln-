static void smbios_build_type_1_fields(QemuOpts *opts)

{

    const char *val;



    val = qemu_opt_get(opts, "manufacturer");

    if (val) {

        smbios_add_field(1, offsetof(struct smbios_type_1, manufacturer_str),

                         val, strlen(val) + 1);

    }

    val = qemu_opt_get(opts, "product");

    if (val) {

        smbios_add_field(1, offsetof(struct smbios_type_1, product_name_str),

                         val, strlen(val) + 1);

    }

    val = qemu_opt_get(opts, "version");

    if (val) {

        smbios_add_field(1, offsetof(struct smbios_type_1, version_str),

                         val, strlen(val) + 1);

    }

    val = qemu_opt_get(opts, "serial");

    if (val) {

        smbios_add_field(1, offsetof(struct smbios_type_1, serial_number_str),

                         val, strlen(val) + 1);

    }

    val = qemu_opt_get(opts, "uuid");

    if (val) {

        if (qemu_uuid_parse(val, qemu_uuid) != 0) {

            error_report("Invalid UUID");

            exit(1);

        }

    }

    val = qemu_opt_get(opts, "sku");

    if (val) {

        smbios_add_field(1, offsetof(struct smbios_type_1, sku_number_str),

                         val, strlen(val) + 1);

    }

    val = qemu_opt_get(opts, "family");

    if (val) {

        smbios_add_field(1, offsetof(struct smbios_type_1, family_str),

                         val, strlen(val) + 1);

    }

}
