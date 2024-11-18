TPMVersion tpm_backend_get_tpm_version(TPMBackend *s)

{

    TPMBackendClass *k = TPM_BACKEND_GET_CLASS(s);



    assert(k->get_tpm_version);



    return k->get_tpm_version(s);

}
