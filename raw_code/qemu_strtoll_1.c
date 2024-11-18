int qemu_strtoll(const char *nptr, const char **endptr, int base,

                 int64_t *result)

{

    char *p;

    int err = 0;

    if (!nptr) {

        if (endptr) {

            *endptr = nptr;

        }

        err = -EINVAL;

    } else {

        errno = 0;

        *result = strtoll(nptr, &p, base);

        err = check_strtox_error(endptr, p, errno);

    }

    return err;

}
