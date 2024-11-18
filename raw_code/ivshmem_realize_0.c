static void ivshmem_realize(PCIDevice *dev, Error **errp)

{

    IVShmemState *s = IVSHMEM_COMMON(dev);



    if (!qtest_enabled()) {

        error_report("ivshmem is deprecated, please use ivshmem-plain"

                     " or ivshmem-doorbell instead");

    }



    if (!!qemu_chr_fe_get_driver(&s->server_chr) + !!s->shmobj != 1) {

        error_setg(errp, "You must specify either 'shm' or 'chardev'");

        return;

    }



    if (s->sizearg == NULL) {

        s->legacy_size = 4 << 20; /* 4 MB default */

    } else {

        int64_t size = qemu_strtosz_MiB(s->sizearg, NULL);

        if (size < 0 || (size_t)size != size || !is_power_of_2(size)) {

            error_setg(errp, "Invalid size %s", s->sizearg);

            return;

        }

        s->legacy_size = size;

    }



    /* check that role is reasonable */

    if (s->role) {

        if (strncmp(s->role, "peer", 5) == 0) {

            s->master = ON_OFF_AUTO_OFF;

        } else if (strncmp(s->role, "master", 7) == 0) {

            s->master = ON_OFF_AUTO_ON;

        } else {

            error_setg(errp, "'role' must be 'peer' or 'master'");

            return;

        }

    } else {

        s->master = ON_OFF_AUTO_AUTO;

    }



    if (s->shmobj) {

        desugar_shm(s);

    }



    /*

     * Note: we don't use INTx with IVSHMEM_MSI at all, so this is a

     * bald-faced lie then.  But it's a backwards compatible lie.

     */

    pci_config_set_interrupt_pin(dev->config, 1);



    ivshmem_common_realize(dev, errp);

}
