int qemu_strtoul(const char *nptr, const char **endptr, int base,

                 unsigned long *result)

{

    char *ep;

    int err = 0;

    if (!nptr) {

        if (endptr) {

            *endptr = nptr;

        }

        err = -EINVAL;

    } else {

        errno = 0;

        *result = strtoul(nptr, &ep, base);

        /* Windows returns 1 for negative out-of-range values.  */

        if (errno == ERANGE) {

            *result = -1;

        }

        err = check_strtox_error(nptr, ep, endptr, errno);

    }

    return err;

}
