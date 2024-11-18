void tpm_backend_cancel_cmd(TPMBackend *s)

{

    TPMBackendClass *k = TPM_BACKEND_GET_CLASS(s);



    assert(k->cancel_cmd);



    k->cancel_cmd(s);

}
