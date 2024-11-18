static int pci_ivshmem_init(PCIDevice *dev)

{

    IVShmemState *s = DO_UPCAST(IVShmemState, dev, dev);

    uint8_t *pci_conf;



    if (s->sizearg == NULL)

        s->ivshmem_size = 4 << 20; /* 4 MB default */

    else {

        s->ivshmem_size = ivshmem_get_size(s);

    }



    register_savevm(&s->dev.qdev, "ivshmem", 0, 0, ivshmem_save, ivshmem_load,

                                                                        dev);



    /* IRQFD requires MSI */

    if (ivshmem_has_feature(s, IVSHMEM_IOEVENTFD) &&

        !ivshmem_has_feature(s, IVSHMEM_MSI)) {

        fprintf(stderr, "ivshmem: ioeventfd/irqfd requires MSI\n");

        exit(1);

    }



    /* check that role is reasonable */

    if (s->role) {

        if (strncmp(s->role, "peer", 5) == 0) {

            s->role_val = IVSHMEM_PEER;

        } else if (strncmp(s->role, "master", 7) == 0) {

            s->role_val = IVSHMEM_MASTER;

        } else {

            fprintf(stderr, "ivshmem: 'role' must be 'peer' or 'master'\n");

            exit(1);

        }

    } else {

        s->role_val = IVSHMEM_MASTER; /* default */

    }



    if (s->role_val == IVSHMEM_PEER) {

        register_device_unmigratable(&s->dev.qdev, "ivshmem", s);

    }



    pci_conf = s->dev.config;

    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_REDHAT_QUMRANET);

    pci_conf[0x02] = 0x10;

    pci_conf[0x03] = 0x11;

    pci_conf[PCI_COMMAND] = PCI_COMMAND_IO | PCI_COMMAND_MEMORY;

    pci_config_set_class(pci_conf, PCI_CLASS_MEMORY_RAM);

    pci_conf[PCI_HEADER_TYPE] = PCI_HEADER_TYPE_NORMAL;



    pci_config_set_interrupt_pin(pci_conf, 1);



    s->shm_pci_addr = 0;

    s->ivshmem_offset = 0;

    s->shm_fd = 0;



    s->ivshmem_mmio_io_addr = cpu_register_io_memory(ivshmem_mmio_read,

                                    ivshmem_mmio_write, s, DEVICE_NATIVE_ENDIAN);

    /* region for registers*/

    pci_register_bar(&s->dev, 0, IVSHMEM_REG_BAR_SIZE,

                           PCI_BASE_ADDRESS_SPACE_MEMORY, ivshmem_mmio_map);



    if ((s->server_chr != NULL) &&

                        (strncmp(s->server_chr->filename, "unix:", 5) == 0)) {

        /* if we get a UNIX socket as the parameter we will talk

         * to the ivshmem server to receive the memory region */



        if (s->shmobj != NULL) {

            fprintf(stderr, "WARNING: do not specify both 'chardev' "

                                                "and 'shm' with ivshmem\n");

        }



        IVSHMEM_DPRINTF("using shared memory server (socket = %s)\n",

                                                    s->server_chr->filename);



        if (ivshmem_has_feature(s, IVSHMEM_MSI)) {

            ivshmem_setup_msi(s);

        }



        /* we allocate enough space for 16 guests and grow as needed */

        s->nb_peers = 16;

        s->vm_id = -1;



        /* allocate/initialize space for interrupt handling */

        s->peers = qemu_mallocz(s->nb_peers * sizeof(Peer));



        pci_register_bar(&s->dev, 2, s->ivshmem_size,

                                PCI_BASE_ADDRESS_SPACE_MEMORY, ivshmem_map);



        s->eventfd_chr = qemu_mallocz(s->vectors * sizeof(CharDriverState *));



        qemu_chr_add_handlers(s->server_chr, ivshmem_can_receive, ivshmem_read,

                     ivshmem_event, s);

    } else {

        /* just map the file immediately, we're not using a server */

        int fd;



        if (s->shmobj == NULL) {

            fprintf(stderr, "Must specify 'chardev' or 'shm' to ivshmem\n");

        }



        IVSHMEM_DPRINTF("using shm_open (shm object = %s)\n", s->shmobj);



        /* try opening with O_EXCL and if it succeeds zero the memory

         * by truncating to 0 */

        if ((fd = shm_open(s->shmobj, O_CREAT|O_RDWR|O_EXCL,

                        S_IRWXU|S_IRWXG|S_IRWXO)) > 0) {

           /* truncate file to length PCI device's memory */

            if (ftruncate(fd, s->ivshmem_size) != 0) {

                fprintf(stderr, "ivshmem: could not truncate shared file\n");

            }



        } else if ((fd = shm_open(s->shmobj, O_CREAT|O_RDWR,

                        S_IRWXU|S_IRWXG|S_IRWXO)) < 0) {

            fprintf(stderr, "ivshmem: could not open shared file\n");

            exit(-1);



        }



        if (check_shm_size(s, fd) == -1) {

            exit(-1);

        }



        create_shared_memory_BAR(s, fd);



    }



    return 0;

}
