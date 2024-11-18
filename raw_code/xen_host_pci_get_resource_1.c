static int xen_host_pci_get_resource(XenHostPCIDevice *d)

{

    int i, rc, fd;

    char path[PATH_MAX];

    char buf[XEN_HOST_PCI_RESOURCE_BUFFER_SIZE];

    unsigned long long start, end, flags, size;

    char *endptr, *s;

    uint8_t type;



    rc = xen_host_pci_sysfs_path(d, "resource", path, sizeof (path));

    if (rc) {

        return rc;

    }

    fd = open(path, O_RDONLY);

    if (fd == -1) {

        XEN_HOST_PCI_LOG("Error: Can't open %s: %s\n", path, strerror(errno));

        return -errno;

    }



    do {

        rc = read(fd, &buf, sizeof (buf) - 1);

        if (rc < 0 && errno != EINTR) {

            rc = -errno;

            goto out;

        }

    } while (rc < 0);

    buf[rc] = 0;

    rc = 0;



    s = buf;

    for (i = 0; i < PCI_NUM_REGIONS; i++) {

        type = 0;



        start = strtoll(s, &endptr, 16);

        if (*endptr != ' ' || s == endptr) {

            break;

        }

        s = endptr + 1;

        end = strtoll(s, &endptr, 16);

        if (*endptr != ' ' || s == endptr) {

            break;

        }

        s = endptr + 1;

        flags = strtoll(s, &endptr, 16);

        if (*endptr != '\n' || s == endptr) {

            break;

        }

        s = endptr + 1;



        if (start) {

            size = end - start + 1;

        } else {

            size = 0;

        }



        if (flags & IORESOURCE_IO) {

            type |= XEN_HOST_PCI_REGION_TYPE_IO;

        }

        if (flags & IORESOURCE_MEM) {

            type |= XEN_HOST_PCI_REGION_TYPE_MEM;

        }

        if (flags & IORESOURCE_PREFETCH) {

            type |= XEN_HOST_PCI_REGION_TYPE_PREFETCH;

        }

        if (flags & IORESOURCE_MEM_64) {

            type |= XEN_HOST_PCI_REGION_TYPE_MEM_64;

        }



        if (i < PCI_ROM_SLOT) {

            d->io_regions[i].base_addr = start;

            d->io_regions[i].size = size;

            d->io_regions[i].type = type;

            d->io_regions[i].bus_flags = flags & IORESOURCE_BITS;

        } else {

            d->rom.base_addr = start;

            d->rom.size = size;

            d->rom.type = type;

            d->rom.bus_flags = flags & IORESOURCE_BITS;

        }

    }

    if (i != PCI_NUM_REGIONS) {

        /* Invalid format or input to short */

        rc = -ENODEV;

    }



out:

    close(fd);

    return rc;

}
