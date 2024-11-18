int ppc_find_by_name (const unsigned char *name, ppc_def_t **def)

{

    int i, ret;



    ret = -1;

    *def = NULL;

    for (i = 0; strcmp(ppc_defs[i].name, "default") != 0; i++) {

        if (strcasecmp(name, ppc_defs[i].name) == 0) {

            *def = &ppc_defs[i];

            ret = 0;

            break;

        }

    }



    return ret;

}
