static void configure_rtc(QemuOpts *opts)

{

    const char *value;



    value = qemu_opt_get(opts, "base");

    if (value) {

        if (!strcmp(value, "utc")) {

            rtc_utc = 1;

        } else if (!strcmp(value, "localtime")) {

            rtc_utc = 0;

        } else {

            configure_rtc_date_offset(value, 0);

        }

    }

    value = qemu_opt_get(opts, "clock");

    if (value) {

        if (!strcmp(value, "host")) {

            rtc_clock = QEMU_CLOCK_HOST;

        } else if (!strcmp(value, "rt")) {

            rtc_clock = QEMU_CLOCK_REALTIME;

        } else if (!strcmp(value, "vm")) {

            rtc_clock = QEMU_CLOCK_VIRTUAL;

        } else {

            fprintf(stderr, "qemu: invalid option value '%s'\n", value);

            exit(1);

        }

    }

    value = qemu_opt_get(opts, "driftfix");

    if (value) {

        if (!strcmp(value, "slew")) {

            static GlobalProperty slew_lost_ticks[] = {

                {

                    .driver   = "mc146818rtc",

                    .property = "lost_tick_policy",

                    .value    = "slew",

                },

                { /* end of list */ }

            };



            qdev_prop_register_global_list(slew_lost_ticks);

        } else if (!strcmp(value, "none")) {

            /* discard is default */

        } else {

            fprintf(stderr, "qemu: invalid option value '%s'\n", value);

            exit(1);

        }

    }

}
