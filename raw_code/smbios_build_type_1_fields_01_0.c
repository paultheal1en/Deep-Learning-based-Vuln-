static void smbios_build_type_1_fields(void)

{

    smbios_maybe_add_str(1, offsetof(struct smbios_type_1, manufacturer_str),

                         type1.manufacturer);

    smbios_maybe_add_str(1, offsetof(struct smbios_type_1, product_name_str),

                         type1.product);

    smbios_maybe_add_str(1, offsetof(struct smbios_type_1, version_str),

                         type1.version);

    smbios_maybe_add_str(1, offsetof(struct smbios_type_1, serial_number_str),

                         type1.serial);

    smbios_maybe_add_str(1, offsetof(struct smbios_type_1, sku_number_str),

                         type1.sku);

    smbios_maybe_add_str(1, offsetof(struct smbios_type_1, family_str),

                         type1.family);

    if (qemu_uuid_set) {

        /* We don't encode the UUID in the "wire format" here because this

         * function is for legacy mode and needs to keep the guest ABI, and

         * because we don't know what's the SMBIOS version advertised by the

         * BIOS.

         */

        smbios_add_field(1, offsetof(struct smbios_type_1, uuid),

                         qemu_uuid, 16);

    }

}
