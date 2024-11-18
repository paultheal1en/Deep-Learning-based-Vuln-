void qemu_check_nic_model(NICInfo *nd, const char *model)

{

    const char *models[2];



    models[0] = model;

    models[1] = NULL;



    qemu_check_nic_model_list(nd, models, model);

}
