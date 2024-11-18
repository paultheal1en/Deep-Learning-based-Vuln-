static void handle_arg_reserved_va(const char *arg)

{

    char *p;

    int shift = 0;

    reserved_va = strtoul(arg, &p, 0);

    switch (*p) {

    case 'k':

    case 'K':

        shift = 10;

        break;

    case 'M':

        shift = 20;

        break;

    case 'G':

        shift = 30;

        break;

    }

    if (shift) {

        unsigned long unshifted = reserved_va;

        p++;

        reserved_va <<= shift;

        if (((reserved_va >> shift) != unshifted)

#if HOST_LONG_BITS > TARGET_VIRT_ADDR_SPACE_BITS

            || (reserved_va > (1ul << TARGET_VIRT_ADDR_SPACE_BITS))

#endif

            ) {

            fprintf(stderr, "Reserved virtual address too big\n");

            exit(EXIT_FAILURE);

        }

    }

    if (*p) {

        fprintf(stderr, "Unrecognised -R size suffix '%s'\n", p);

        exit(EXIT_FAILURE);

    }

}
