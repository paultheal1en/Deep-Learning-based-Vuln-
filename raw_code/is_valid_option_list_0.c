bool is_valid_option_list(const char *param)

{

    size_t buflen = strlen(param) + 1;

    char *buf = g_malloc(buflen);

    const char *p = param;

    bool result = true;



    while (*p) {

        p = get_opt_value(buf, buflen, p);

        if (*p && !*++p) {

            result = false;

            goto out;

        }



        if (!*buf || *buf == ',') {

            result = false;

            goto out;

        }

    }



out:

    free(buf);

    return result;

}
