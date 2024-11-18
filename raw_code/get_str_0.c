static int get_str(char *buf, int buf_size, const char **pp)

{

    const char *p;

    char *q;

    int c;



    q = buf;

    p = *pp;

    while (qemu_isspace(*p))

        p++;

    if (*p == '\0') {

    fail:

        *q = '\0';

        *pp = p;

        return -1;

    }

    if (*p == '\"') {

        p++;

        while (*p != '\0' && *p != '\"') {

            if (*p == '\\') {

                p++;

                c = *p++;

                switch(c) {

                case 'n':

                    c = '\n';

                    break;

                case 'r':

                    c = '\r';

                    break;

                case '\\':

                case '\'':

                case '\"':

                    break;

                default:

                    qemu_printf("unsupported escape code: '\\%c'\n", c);

                    goto fail;

                }

                if ((q - buf) < buf_size - 1) {

                    *q++ = c;

                }

            } else {

                if ((q - buf) < buf_size - 1) {

                    *q++ = *p;

                }

                p++;

            }

        }

        if (*p != '\"') {

            qemu_printf("unterminated string\n");

            goto fail;

        }

        p++;

    } else {

        while (*p != '\0' && !qemu_isspace(*p)) {

            if ((q - buf) < buf_size - 1) {

                *q++ = *p;

            }

            p++;

        }

    }

    *q = '\0';

    *pp = p;

    return 0;

}
