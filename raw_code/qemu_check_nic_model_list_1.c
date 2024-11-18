int qemu_check_nic_model_list(NICInfo *nd, const char * const *models,

                              const char *default_model)

{

    int i, exit_status = 0;



    if (!nd->model)

        nd->model = strdup(default_model);



    if (strcmp(nd->model, "?") != 0) {

        for (i = 0 ; models[i]; i++)

            if (strcmp(nd->model, models[i]) == 0)

                return i;



        fprintf(stderr, "qemu: Unsupported NIC model: %s\n", nd->model);

        exit_status = 1;

    }



    fprintf(stderr, "qemu: Supported NIC models: ");

    for (i = 0 ; models[i]; i++)

        fprintf(stderr, "%s%c", models[i], models[i+1] ? ',' : '\n');



    exit(exit_status);

}
