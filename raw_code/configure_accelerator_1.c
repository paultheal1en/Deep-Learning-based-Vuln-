static int configure_accelerator(void)

{

    const char *p = NULL;

    char buf[10];

    int i, ret;

    bool accel_initialised = false;

    bool init_failed = false;



    QemuOptsList *list = qemu_find_opts("machine");

    if (!QTAILQ_EMPTY(&list->head)) {

        p = qemu_opt_get(QTAILQ_FIRST(&list->head), "accel");

    }



    if (p == NULL) {

        /* Use the default "accelerator", tcg */

        p = "tcg";

    }



    while (!accel_initialised && *p != '\0') {

        if (*p == ':') {

            p++;

        }

        p = get_opt_name(buf, sizeof (buf), p, ':');

        for (i = 0; i < ARRAY_SIZE(accel_list); i++) {

            if (strcmp(accel_list[i].opt_name, buf) == 0) {

                if (!accel_list[i].available()) {

                    printf("%s not supported for this target\n",

                           accel_list[i].name);

                    continue;

                }

                *(accel_list[i].allowed) = true;

                ret = accel_list[i].init();

                if (ret < 0) {

                    init_failed = true;

                    fprintf(stderr, "failed to initialize %s: %s\n",

                            accel_list[i].name,

                            strerror(-ret));

                    *(accel_list[i].allowed) = false;

                } else {

                    accel_initialised = true;

                }

                break;

            }

        }

        if (i == ARRAY_SIZE(accel_list)) {

            fprintf(stderr, "\"%s\" accelerator does not exist.\n", buf);

        }

    }



    if (!accel_initialised) {

        if (!init_failed) {

            fprintf(stderr, "No accelerator found!\n");

        }

        exit(1);

    }



    if (init_failed) {

        fprintf(stderr, "Back to %s accelerator.\n", accel_list[i].name);

    }



    return !accel_initialised;

}
