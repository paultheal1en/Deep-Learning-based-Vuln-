static void allocate_system_memory_nonnuma(MemoryRegion *mr, Object *owner,

                                           const char *name,

                                           uint64_t ram_size)

{

    if (mem_path) {

#ifdef __linux__

        Error *err = NULL;

        memory_region_init_ram_from_file(mr, owner, name, ram_size, false,

                                         mem_path, &err);



        /* Legacy behavior: if allocation failed, fall back to

         * regular RAM allocation.

         */

        if (err) {

            error_report_err(err);

            memory_region_init_ram(mr, owner, name, ram_size, &error_abort);

        }

#else

        fprintf(stderr, "-mem-path not supported on this host\n");

        exit(1);

#endif

    } else {

        memory_region_init_ram(mr, owner, name, ram_size, &error_abort);

    }

    vmstate_register_ram_global(mr);

}
