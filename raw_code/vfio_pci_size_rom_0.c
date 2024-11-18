static void vfio_pci_size_rom(VFIOPCIDevice *vdev)

{

    uint32_t orig, size = cpu_to_le32((uint32_t)PCI_ROM_ADDRESS_MASK);

    off_t offset = vdev->config_offset + PCI_ROM_ADDRESS;

    DeviceState *dev = DEVICE(vdev);

    char name[32];

    int fd = vdev->vbasedev.fd;



    if (vdev->pdev.romfile || !vdev->pdev.rom_bar) {

        /* Since pci handles romfile, just print a message and return */

        if (vfio_blacklist_opt_rom(vdev) && vdev->pdev.romfile) {

            error_printf("Warning : Device at %04x:%02x:%02x.%x "

                         "is known to cause system instability issues during "

                         "option rom execution. "

                         "Proceeding anyway since user specified romfile\n",

                         vdev->host.domain, vdev->host.bus, vdev->host.slot,

                         vdev->host.function);

        }

        return;

    }



    /*

     * Use the same size ROM BAR as the physical device.  The contents

     * will get filled in later when the guest tries to read it.

     */

    if (pread(fd, &orig, 4, offset) != 4 ||

        pwrite(fd, &size, 4, offset) != 4 ||

        pread(fd, &size, 4, offset) != 4 ||

        pwrite(fd, &orig, 4, offset) != 4) {

        error_report("%s(%04x:%02x:%02x.%x) failed: %m",

                     __func__, vdev->host.domain, vdev->host.bus,

                     vdev->host.slot, vdev->host.function);

        return;

    }



    size = ~(le32_to_cpu(size) & PCI_ROM_ADDRESS_MASK) + 1;



    if (!size) {

        return;

    }



    if (vfio_blacklist_opt_rom(vdev)) {

        if (dev->opts && qemu_opt_get(dev->opts, "rombar")) {

            error_printf("Warning : Device at %04x:%02x:%02x.%x "

                         "is known to cause system instability issues during "

                         "option rom execution. "

                         "Proceeding anyway since user specified non zero value for "

                         "rombar\n",

                         vdev->host.domain, vdev->host.bus, vdev->host.slot,

                         vdev->host.function);

        } else {

            error_printf("Warning : Rom loading for device at "

                         "%04x:%02x:%02x.%x has been disabled due to "

                         "system instability issues. "

                         "Specify rombar=1 or romfile to force\n",

                         vdev->host.domain, vdev->host.bus, vdev->host.slot,

                         vdev->host.function);

            return;

        }

    }



    trace_vfio_pci_size_rom(vdev->vbasedev.name, size);



    snprintf(name, sizeof(name), "vfio[%04x:%02x:%02x.%x].rom",

             vdev->host.domain, vdev->host.bus, vdev->host.slot,

             vdev->host.function);



    memory_region_init_io(&vdev->pdev.rom, OBJECT(vdev),

                          &vfio_rom_ops, vdev, name, size);



    pci_register_bar(&vdev->pdev, PCI_ROM_SLOT,

                     PCI_BASE_ADDRESS_SPACE_MEMORY, &vdev->pdev.rom);



    vdev->pdev.has_rom = true;

    vdev->rom_read_failed = false;

}
