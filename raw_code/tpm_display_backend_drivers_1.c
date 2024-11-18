static void tpm_display_backend_drivers(void)

{

    int i;



    fprintf(stderr, "Supported TPM types (choose only one):\n");



    for (i = 0; i < TPM_MAX_DRIVERS && be_drivers[i] != NULL; i++) {

        fprintf(stderr, "%12s   %s\n",

                TpmType_lookup[be_drivers[i]->type], be_drivers[i]->desc());

    }

    fprintf(stderr, "\n");

}
