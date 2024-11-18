static TPMInfo *qmp_query_tpm_inst(TPMBackend *drv)

{

    TPMInfo *res = g_new0(TPMInfo, 1);

    TPMPassthroughOptions *tpo;



    res->id = g_strdup(drv->id);

    res->model = drv->fe_model;

    res->options = g_new0(TpmTypeOptions, 1);



    switch (drv->ops->type) {

    case TPM_TYPE_PASSTHROUGH:

        res->options->type = TPM_TYPE_OPTIONS_KIND_PASSTHROUGH;

        tpo = g_new0(TPMPassthroughOptions, 1);

        res->options->u.passthrough = tpo;

        if (drv->path) {

            tpo->path = g_strdup(drv->path);

            tpo->has_path = true;

        }

        if (drv->cancel_path) {

            tpo->cancel_path = g_strdup(drv->cancel_path);

            tpo->has_cancel_path = true;

        }

        break;

    case TPM_TYPE__MAX:

        break;

    }



    return res;

}
