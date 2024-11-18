static const TPMDriverOps *tpm_driver_find_by_type(enum TpmType type)

{

    int i;



    for (i = 0; i < TPM_MAX_DRIVERS && be_drivers[i] != NULL; i++) {

        if (be_drivers[i]->type == type) {

            return be_drivers[i];

        }

    }

    return NULL;

}
