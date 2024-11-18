int check_params(const char * const *params, const char *str)

{

    int name_buf_size = 1;

    const char *p;

    char *name_buf;

    int i, len;

    int ret = 0;



    for (i = 0; params[i] != NULL; i++) {

        len = strlen(params[i]) + 1;

        if (len > name_buf_size) {

            name_buf_size = len;

        }

    }

    name_buf = qemu_malloc(name_buf_size);



    p = str;

    while (*p != '\0') {

        p = get_opt_name(name_buf, name_buf_size, p, '=');

        if (*p != '=') {

            ret = -1;

            break;

        }

        p++;

        for(i = 0; params[i] != NULL; i++)

            if (!strcmp(params[i], name_buf))

                break;

        if (params[i] == NULL) {

            ret = -1;

            break;

        }

        p = get_opt_value(NULL, 0, p);

        if (*p != ',')

            break;

        p++;

    }



    qemu_free(name_buf);

    return ret;

}
