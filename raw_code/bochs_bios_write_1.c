static void bochs_bios_write(void *opaque, uint32_t addr, uint32_t val)

{

    static const char shutdown_str[8] = "Shutdown";

    static int shutdown_index = 0;



    switch(addr) {

        /* Bochs BIOS messages */

    case 0x400:

    case 0x401:

        /* used to be panic, now unused */

        break;

    case 0x402:

    case 0x403:

#ifdef DEBUG_BIOS

        fprintf(stderr, "%c", val);

#endif

        break;

    case 0x8900:

        /* same as Bochs power off */

        if (val == shutdown_str[shutdown_index]) {

            shutdown_index++;

            if (shutdown_index == 8) {

                shutdown_index = 0;

                qemu_system_shutdown_request();

            }

        } else {

            shutdown_index = 0;

        }

        break;



        /* LGPL'ed VGA BIOS messages */

    case 0x501:

    case 0x502:

        fprintf(stderr, "VGA BIOS panic, line %d\n", val);

        exit(1);

    case 0x500:

    case 0x503:

#ifdef DEBUG_BIOS

        fprintf(stderr, "%c", val);

#endif

        break;

    }

}
