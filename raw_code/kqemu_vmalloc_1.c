void *kqemu_vmalloc(size_t size)

{

    static int phys_ram_fd = -1;

    static int phys_ram_size = 0;

    const char *tmpdir;

    char phys_ram_file[1024];

    void *ptr;

    struct statfs stfs;



    if (phys_ram_fd < 0) {

        tmpdir = getenv("QEMU_TMPDIR");

        if (!tmpdir)

            tmpdir = "/dev/shm";

        if (statfs(tmpdir, &stfs) == 0) {

            int64_t free_space;

            int ram_mb;



            extern int ram_size;

            free_space = (int64_t)stfs.f_bavail * stfs.f_bsize;

            if ((ram_size + 8192 * 1024) >= free_space) {

                ram_mb = (ram_size / (1024 * 1024));

                fprintf(stderr, 

                        "You do not have enough space in '%s' for the %d MB of QEMU virtual RAM.\n",

                        tmpdir, ram_mb);

                if (strcmp(tmpdir, "/dev/shm") == 0) {

                    fprintf(stderr, "To have more space available provided you have enough RAM and swap, do as root:\n"

                            "umount /dev/shm\n"

                            "mount -t tmpfs -o size=%dm none /dev/shm\n",

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

        if (mkstemp(phys_ram_file) < 0) {

            fprintf(stderr, 

                    "warning: could not create temporary file in '%s'.\n"

                    "Use QEMU_TMPDIR to select a directory in a tmpfs filesystem.\n"

                    "Using '/tmp' as fallback.\n",

                    tmpdir);

            snprintf(phys_ram_file, sizeof(phys_ram_file), "%s/qemuXXXXXX", 

                     "/tmp");

            if (mkstemp(phys_ram_file) < 0) {

                fprintf(stderr, "Could not create temporary memory file '%s'\n", 

                        phys_ram_file);

                exit(1);

            }

        }

        phys_ram_fd = open(phys_ram_file, O_CREAT | O_TRUNC | O_RDWR, 0600);

        if (phys_ram_fd < 0) {

            fprintf(stderr, "Could not open temporary memory file '%s'\n", 

                    phys_ram_file);

            exit(1);

        }

        unlink(phys_ram_file);

    }

    size = (size + 4095) & ~4095;

    ftruncate(phys_ram_fd, phys_ram_size + size);

    ptr = mmap(NULL, 

               size, 

               PROT_WRITE | PROT_READ, MAP_SHARED, 

               phys_ram_fd, phys_ram_size);

    if (ptr == MAP_FAILED) {

        fprintf(stderr, "Could not map physical memory\n");

        exit(1);

    }

    phys_ram_size += size;

    return ptr;

}
