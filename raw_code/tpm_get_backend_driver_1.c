const TPMDriverOps *tpm_get_backend_driver(const char *type)

{

    int i;



    for (i = 0; i < TPM_MAX_DRIVERS && be_drivers[i] != NULL; i++) {

        if (!strcmp(TpmType_lookup[be_drivers[i]->type], type)) {

            return be_drivers[i];

        }

    }



    return NULL;

}
