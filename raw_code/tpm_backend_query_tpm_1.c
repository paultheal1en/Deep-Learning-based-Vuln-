TPMInfo *tpm_backend_query_tpm(TPMBackend *s)

{

    TPMInfo *info = g_new0(TPMInfo, 1);

    TPMBackendClass *k = TPM_BACKEND_GET_CLASS(s);

    TPMIfClass *tic = TPM_IF_GET_CLASS(s->tpmif);



    info->id = g_strdup(s->id);

    info->model = tic->model;

    if (k->get_tpm_options) {

        info->options = k->get_tpm_options(s);

    }



    return info;

}
