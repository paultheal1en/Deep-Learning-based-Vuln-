static const OptionDef *find_option(const OptionDef *po, const char *name)

{

    const char *p = strchr(name, ':');

    int len = p ? p - name : strlen(name);



    while (po->name != NULL) {

        if (!strncmp(name, po->name, len) && strlen(po->name) == len)

            break;

        po++;

    }

    return po;

}
