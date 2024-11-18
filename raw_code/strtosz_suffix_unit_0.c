int64_t strtosz_suffix_unit(const char *nptr, char **end,

                            const char default_suffix, int64_t unit)

{

    int64_t retval = -1;

    char *endptr;

    unsigned char c, d;

    int mul_required = 0;

    double val, mul, integral, fraction;



    errno = 0;

    val = strtod(nptr, &endptr);

    if (isnan(val) || endptr == nptr || errno != 0) {

        goto fail;

    }

    fraction = modf(val, &integral);

    if (fraction != 0) {

        mul_required = 1;

    }

    /*

     * Any whitespace character is fine for terminating the number,

     * in addition we accept ',' to handle strings where the size is

     * part of a multi token argument.

     */

    c = *endptr;

    d = c;

    if (qemu_isspace(c) || c == '\0' || c == ',') {

        c = 0;

        d = default_suffix;

    }

    switch (qemu_toupper(d)) {

    case STRTOSZ_DEFSUFFIX_B:

        mul = 1;

        if (mul_required) {

            goto fail;

        }

        break;

    case STRTOSZ_DEFSUFFIX_KB:

        mul = unit;

        break;

    case STRTOSZ_DEFSUFFIX_MB:

        mul = unit * unit;

        break;

    case STRTOSZ_DEFSUFFIX_GB:

        mul = unit * unit * unit;

        break;

    case STRTOSZ_DEFSUFFIX_TB:

        mul = unit * unit * unit * unit;

        break;

    default:

        goto fail;

    }

    /*

     * If not terminated by whitespace, ',', or \0, increment endptr

     * to point to next character, then check that we are terminated

     * by an appropriate separating character, ie. whitespace, ',', or

     * \0. If not, we are seeing trailing garbage, thus fail.

     */

    if (c != 0) {

        endptr++;

        if (!qemu_isspace(*endptr) && *endptr != ',' && *endptr != 0) {

            goto fail;

        }

    }

    if ((val * mul >= INT64_MAX) || val < 0) {

        goto fail;

    }

    retval = val * mul;



fail:

    if (end) {

        *end = endptr;

    }



    return retval;

}
