int select_watchdog(const char *p)

{

    WatchdogTimerModel *model;

    QemuOpts *opts;



    /* -watchdog ? lists available devices and exits cleanly. */

    if (strcmp(p, "?") == 0) {

        LIST_FOREACH(model, &watchdog_list, entry) {

            fprintf(stderr, "\t%s\t%s\n",

                     model->wdt_name, model->wdt_description);

        }

        return 2;

    }



    LIST_FOREACH(model, &watchdog_list, entry) {

        if (strcasecmp(model->wdt_name, p) == 0) {

            /* add the device */

            opts = qemu_opts_create(&qemu_device_opts, NULL, 0);

            qemu_opt_set(opts, "driver", p);

            return 0;

        }

    }



    fprintf(stderr, "Unknown -watchdog device. Supported devices are:\n");

    LIST_FOREACH(model, &watchdog_list, entry) {

        fprintf(stderr, "\t%s\t%s\n",

                 model->wdt_name, model->wdt_description);

    }

    return 1;

}
