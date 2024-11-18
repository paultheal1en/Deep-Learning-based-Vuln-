static CharDriverState *qemu_chr_open_spice_vmc(const char *id,

                                                ChardevBackend *backend,

                                                ChardevReturn *ret,

                                                Error **errp)

{

    const char *type = backend->u.spicevmc->type;

    const char **psubtype = spice_server_char_device_recognized_subtypes();



    for (; *psubtype != NULL; ++psubtype) {

        if (strcmp(type, *psubtype) == 0) {

            break;

        }

    }

    if (*psubtype == NULL) {

        fprintf(stderr, "spice-qemu-char: unsupported type: %s\n", type);

        print_allowed_subtypes();

        return NULL;

    }



    return chr_open(type, spice_vmc_set_fe_open);

}
