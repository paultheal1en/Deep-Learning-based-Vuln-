static int monitor_parse(const char *devname)

{

    static int index = 0;

    char label[32];



    if (strcmp(devname, "none") == 0)

        return 0;

    if (index == MAX_MONITOR_DEVICES) {

        fprintf(stderr, "qemu: too many monitor devices\n");

        exit(1);

    }

    if (index == 0) {

        snprintf(label, sizeof(label), "monitor");

    } else {

        snprintf(label, sizeof(label), "monitor%d", index);

    }

    monitor_hds[index] = qemu_chr_open(label, devname, NULL);

    if (!monitor_hds[index]) {

        fprintf(stderr, "qemu: could not open monitor device '%s'\n",

                devname);

        return -1;

    }

    index++;

    return 0;

}
