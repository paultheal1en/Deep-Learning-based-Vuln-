static char *qemu_rbd_next_tok(int max_len,

                               char *src, char delim,

                               const char *name,

                               char **p, Error **errp)

{

    int l;

    char *end;



    *p = NULL;



    if (delim != '\0') {

        for (end = src; *end; ++end) {

            if (*end == delim) {

                break;

            }

            if (*end == '\\' && end[1] != '\0') {

                end++;

            }

        }

        if (*end == delim) {

            *p = end + 1;

            *end = '\0';

        }

    }

    l = strlen(src);

    if (l >= max_len) {

        error_setg(errp, "%s too long", name);

        return NULL;

    } else if (l == 0) {

        error_setg(errp, "%s too short", name);

        return NULL;

    }



    return src;

}
