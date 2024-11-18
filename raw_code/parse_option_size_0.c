void parse_option_size(const char *name, const char *value,

                       uint64_t *ret, Error **errp)

{

    uint64_t size;

    int err;



    err = qemu_strtosz(value, NULL, &size);

    if (err == -ERANGE) {

        error_setg(errp, "Value '%s' is too large for parameter '%s'",

                   value, name);

        return;

    }

    if (err) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE, name,

                   "a non-negative number below 2^64");

        error_append_hint(errp, "Optional suffix k, M, G, T, P or E means"

                          " kilo-, mega-, giga-, tera-, peta-\n"

                          "and exabytes, respectively.\n");

        return;

    }

    *ret = size;

}
