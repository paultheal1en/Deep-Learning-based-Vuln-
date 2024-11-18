char **breakline(char *input, int *count)

{

    int c = 0;

    char *p;

    char **rval = calloc(sizeof(char *), 1);



    while (rval && (p = qemu_strsep(&input, " ")) != NULL) {

        if (!*p) {

            continue;

        }

        c++;

        rval = realloc(rval, sizeof(*rval) * (c + 1));

        if (!rval) {

            c = 0;

            break;

        }

        rval[c - 1] = p;

        rval[c] = NULL;

    }

    *count = c;

    return rval;

}
