static void set_memory_options(uint64_t *ram_slots, ram_addr_t *maxram_size)

{

    uint64_t sz;

    const char *mem_str;

    const char *maxmem_str, *slots_str;

    const ram_addr_t default_ram_size = (ram_addr_t)DEFAULT_RAM_SIZE *

                                        1024 * 1024;

    QemuOpts *opts = qemu_find_opts_singleton("memory");



    sz = 0;

    mem_str = qemu_opt_get(opts, "size");

    if (mem_str) {

        if (!*mem_str) {

            error_report("missing 'size' option value");

            exit(EXIT_FAILURE);

        }



        sz = qemu_opt_get_size(opts, "size", ram_size);



        /* Fix up legacy suffix-less format */

        if (g_ascii_isdigit(mem_str[strlen(mem_str) - 1])) {

            uint64_t overflow_check = sz;



            sz <<= 20;

            if ((sz >> 20) != overflow_check) {

                error_report("too large 'size' option value");

                exit(EXIT_FAILURE);

            }

        }

    }



    /* backward compatibility behaviour for case "-m 0" */

    if (sz == 0) {

        sz = default_ram_size;

    }



    sz = QEMU_ALIGN_UP(sz, 8192);

    ram_size = sz;

    if (ram_size != sz) {

        error_report("ram size too large");

        exit(EXIT_FAILURE);

    }



    /* store value for the future use */

    qemu_opt_set_number(opts, "size", ram_size, &error_abort);

    *maxram_size = ram_size;



    maxmem_str = qemu_opt_get(opts, "maxmem");

    slots_str = qemu_opt_get(opts, "slots");

    if (maxmem_str && slots_str) {

        uint64_t slots;



        sz = qemu_opt_get_size(opts, "maxmem", 0);

        slots = qemu_opt_get_number(opts, "slots", 0);

        if (sz < ram_size) {

            error_report("invalid value of -m option maxmem: "

                         "maximum memory size (0x%" PRIx64 ") must be at least "

                         "the initial memory size (0x" RAM_ADDR_FMT ")",

                         sz, ram_size);

            exit(EXIT_FAILURE);

        } else if (sz > ram_size) {

            if (!slots) {

                error_report("invalid value of -m option: maxmem was "

                             "specified, but no hotplug slots were specified");

                exit(EXIT_FAILURE);

            }

        } else if (slots) {

            error_report("invalid value of -m option maxmem: "

                         "memory slots were specified but maximum memory size "

                         "(0x%" PRIx64 ") is equal to the initial memory size "

                         "(0x" RAM_ADDR_FMT ")", sz, ram_size);

            exit(EXIT_FAILURE);

        }



        *maxram_size = sz;

        *ram_slots = slots;

    } else if ((!maxmem_str && slots_str) ||

            (maxmem_str && !slots_str)) {

        error_report("invalid -m option value: missing "

                "'%s' option", slots_str ? "maxmem" : "slots");

        exit(EXIT_FAILURE);

    }

}
