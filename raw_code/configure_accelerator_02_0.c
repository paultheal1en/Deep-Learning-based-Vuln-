int configure_accelerator(MachineState *ms)

{

    const char *p;

    char buf[10];

    int ret;

    bool accel_initialised = false;

    bool init_failed = false;

    AccelClass *acc = NULL;



    p = qemu_opt_get(qemu_get_machine_opts(), "accel");

    if (p == NULL) {

        /* Use the default "accelerator", tcg */

        p = "tcg";

    }



    while (!accel_initialised && *p != '\0') {

        if (*p == ':') {

            p++;

        }

        p = get_opt_name(buf, sizeof(buf), p, ':');

        acc = accel_find(buf);

        if (!acc) {

            fprintf(stderr, "\"%s\" accelerator not found.\n", buf);

            continue;

        }

        if (acc->available && !acc->available()) {

            printf("%s not supported for this target\n",

                   acc->name);

            continue;

        }

        ret = accel_init_machine(acc, ms);

        if (ret < 0) {

            init_failed = true;

            fprintf(stderr, "failed to initialize %s: %s\n",

                    acc->name,

                    strerror(-ret));

        } else {

            accel_initialised = true;

        }

    }



    if (!accel_initialised) {

        if (!init_failed) {

            fprintf(stderr, "No accelerator found!\n");

        }

        exit(1);

    }



    if (init_failed) {

        fprintf(stderr, "Back to %s accelerator.\n", acc->name);

    }



    return !accel_initialised;

}
