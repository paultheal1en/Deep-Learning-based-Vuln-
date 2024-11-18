static void *kqemu_vmalloc(size_t size)

{

    static int phys_ram_fd = -1;

    static int phys_ram_size = 0;

    void *ptr;



/* no need (?) for a dummy file on OpenBSD/FreeBSD */

#if defined(__OpenBSD__) || defined(__FreeBSD__) || defined(__DragonFly__)

    int map_anon = MAP_ANON;

#else

    int map_anon = 0;

    const char *tmpdir;

    char phys_ram_file[1024];

#ifdef CONFIG_SOLARIS

    struct statvfs stfs;

#else

    struct statfs stfs;

#endif



    if (!size) {

        abort ();

    }



    if (phys_ram_fd < 0) {

        tmpdir = getenv("QEMU_TMPDIR");

        if (!tmpdir)

#ifdef CONFIG_SOLARIS

            tmpdir = "/tmp";

        if (statvfs(tmpdir, &stfs) == 0) {

#else

            tmpdir = "/dev/shm";

        if (statfs(tmpdir, &stfs) == 0) {

#endif

            int64_t free_space;

            int ram_mb;



            free_space = (int64_t)stfs.f_bavail * stfs.f_bsize;

            if ((ram_size + 8192 * 1024) >= free_space) {

                ram_mb = (ram_size / (1024 * 1024));

                fprintf(stderr,

                        "You do not have enough space in '%s' for the %d MB of QEMU virtual RAM.\n",

                        tmpdir, ram_mb);

                if (strcmp(tmpdir, "/dev/shm") == 0) {

                    fprintf(stderr, "To have more space available provided you have enough RAM and swap, do as root:\n"

                            "mount -o remount,size=%dm /dev/shm\n",

                            ram_mb + 16);

                } else {

                    fprintf(stderr,

                            "Use the '-m' option of QEMU to diminish the amount of virtual RAM or use the\n"

                            "QEMU_TMPDIR environment variable to set another directory where the QEMU\n"

                            "temporary RAM file will be opened.\n");

                }

                fprintf(stderr, "Or disable the accelerator module with -no-kqemu\n");

                exit(1);

            }

        }

        snprintf(phys_ram_file, sizeof(phys_ram_file), "%s/qemuXXXXXX",

                 tmpdir);

        phys_ram_fd = mkstemp(phys_ram_file);

        if (phys_ram_fd < 0) {

            fprintf(stderr,

                    "warning: could not create temporary file in '%s'.\n"

                    "Use QEMU_TMPDIR to select a directory in a tmpfs filesystem.\n"

                    "Using '/tmp' as fallback.\n",

                    tmpdir);

            snprintf(phys_ram_file, sizeof(phys_ram_file), "%s/qemuXXXXXX",

                     "/tmp");

            phys_ram_fd = mkstemp(phys_ram_file);

            if (phys_ram_fd < 0) {

                fprintf(stderr, "Could not create temporary memory file '%s'\n",

                        phys_ram_file);

                exit(1);

            }

        }

        unlink(phys_ram_file);

    }

    size = (size + 4095) & ~4095;

    ftruncate(phys_ram_fd, phys_ram_size + size);

#endif /* !(__OpenBSD__ || __FreeBSD__ || __DragonFly__) */

    ptr = mmap(NULL,

               size,

               PROT_WRITE | PROT_READ, map_anon | MAP_SHARED,

               phys_ram_fd, phys_ram_size);

    if (ptr == MAP_FAILED) {

        fprintf(stderr, "Could not map physical memory\n");

        exit(1);

    }

    phys_ram_size += size;

    return ptr;

}
