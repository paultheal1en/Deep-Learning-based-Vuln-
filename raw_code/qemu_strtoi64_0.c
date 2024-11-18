int qemu_strtoi64(const char *nptr, const char **endptr, int base,

                 int64_t *result)

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

        /* FIXME This assumes int64_t is long long */

        *result = strtoll(nptr, &ep, base);

        err = check_strtox_error(nptr, ep, endptr, errno);

    }

    return err;

}
