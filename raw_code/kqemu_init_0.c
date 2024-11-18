int kqemu_init(CPUState *env)

{

    struct kqemu_init kinit;

    int ret, version;

#ifdef _WIN32

    DWORD temp;

#endif



    if (!kqemu_allowed)

        return -1;



#ifdef _WIN32

    kqemu_fd = CreateFile(KQEMU_DEVICE, GENERIC_WRITE | GENERIC_READ,

                          FILE_SHARE_READ | FILE_SHARE_WRITE,

                          NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,

                          NULL);

    if (kqemu_fd == KQEMU_INVALID_FD) {

        fprintf(stderr, "Could not open '%s' - QEMU acceleration layer not activated: %lu\n",

                KQEMU_DEVICE, GetLastError());

        return -1;

    }

#else

    kqemu_fd = open(KQEMU_DEVICE, O_RDWR);

    if (kqemu_fd == KQEMU_INVALID_FD) {

        fprintf(stderr, "Could not open '%s' - QEMU acceleration layer not activated: %s\n",

                KQEMU_DEVICE, strerror(errno));

        return -1;

    }

#endif

    version = 0;

#ifdef _WIN32

    DeviceIoControl(kqemu_fd, KQEMU_GET_VERSION, NULL, 0,

                    &version, sizeof(version), &temp, NULL);

#else

    ioctl(kqemu_fd, KQEMU_GET_VERSION, &version);

#endif

    if (version != KQEMU_VERSION) {

        fprintf(stderr, "Version mismatch between kqemu module and qemu (%08x %08x) - disabling kqemu use\n",

                version, KQEMU_VERSION);

        goto fail;

    }



    pages_to_flush = qemu_vmalloc(KQEMU_MAX_PAGES_TO_FLUSH *

                                  sizeof(uint64_t));

    if (!pages_to_flush)

        goto fail;



    ram_pages_to_update = qemu_vmalloc(KQEMU_MAX_RAM_PAGES_TO_UPDATE *

                                       sizeof(uint64_t));

    if (!ram_pages_to_update)

        goto fail;



    modified_ram_pages = qemu_vmalloc(KQEMU_MAX_MODIFIED_RAM_PAGES *

                                      sizeof(uint64_t));

    if (!modified_ram_pages)

        goto fail;

    modified_ram_pages_table =

        qemu_mallocz(kqemu_phys_ram_size >> TARGET_PAGE_BITS);

    if (!modified_ram_pages_table)

        goto fail;



    memset(&kinit, 0, sizeof(kinit)); /* set the paddings to zero */

    kinit.ram_base = kqemu_phys_ram_base;

    kinit.ram_size = kqemu_phys_ram_size;

    kinit.ram_dirty = phys_ram_dirty;

    kinit.pages_to_flush = pages_to_flush;

    kinit.ram_pages_to_update = ram_pages_to_update;

    kinit.modified_ram_pages = modified_ram_pages;

#ifdef _WIN32

    ret = DeviceIoControl(kqemu_fd, KQEMU_INIT, &kinit, sizeof(kinit),

                          NULL, 0, &temp, NULL) == TRUE ? 0 : -1;

#else

    ret = ioctl(kqemu_fd, KQEMU_INIT, &kinit);

#endif

    if (ret < 0) {

        fprintf(stderr, "Error %d while initializing QEMU acceleration layer - disabling it for now\n", ret);

    fail:

        kqemu_closefd(kqemu_fd);

        kqemu_fd = KQEMU_INVALID_FD;

        return -1;

    }

    kqemu_update_cpuid(env);

    env->kqemu_enabled = kqemu_allowed;

    nb_pages_to_flush = 0;

    nb_ram_pages_to_update = 0;



    qpi_init();

    return 0;

}
