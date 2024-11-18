static int virtcon_parse(const char *devname)

{

    QemuOptsList *device = qemu_find_opts("device");

    static int index = 0;

    char label[32];

    QemuOpts *bus_opts, *dev_opts;



    if (strcmp(devname, "none") == 0)

        return 0;

    if (index == MAX_VIRTIO_CONSOLES) {

        fprintf(stderr, "qemu: too many virtio consoles\n");

        exit(1);

    }



    bus_opts = qemu_opts_create(device, NULL, 0);

    if (arch_type == QEMU_ARCH_S390X) {

        qemu_opt_set(bus_opts, "driver", "virtio-serial-s390");

    } else {

        qemu_opt_set(bus_opts, "driver", "virtio-serial-pci");

    } 



    dev_opts = qemu_opts_create(device, NULL, 0);

    qemu_opt_set(dev_opts, "driver", "virtconsole");



    snprintf(label, sizeof(label), "virtcon%d", index);

    virtcon_hds[index] = qemu_chr_new(label, devname, NULL);

    if (!virtcon_hds[index]) {

        fprintf(stderr, "qemu: could not open virtio console '%s': %s\n",

                devname, strerror(errno));

        return -1;

    }

    qemu_opt_set(dev_opts, "chardev", label);



    index++;

    return 0;

}
