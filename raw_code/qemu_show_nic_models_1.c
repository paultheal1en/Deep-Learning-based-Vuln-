int qemu_show_nic_models(const char *arg, const char *const *models)

{

    int i;



    if (!arg || strcmp(arg, "?"))

        return 0;



    fprintf(stderr, "qemu: Supported NIC models: ");

    for (i = 0 ; models[i]; i++)

        fprintf(stderr, "%s%c", models[i], models[i+1] ? ',' : '\n');

    return 1;

}
