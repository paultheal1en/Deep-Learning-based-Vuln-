static int xen_host_pci_sysfs_path(const XenHostPCIDevice *d,

                                   const char *name, char *buf, ssize_t size)

{

    int rc;



    rc = snprintf(buf, size, "/sys/bus/pci/devices/%04x:%02x:%02x.%d/%s",

                  d->domain, d->bus, d->dev, d->func, name);



    if (rc >= size || rc < 0) {

        /* The output is truncated, or some other error was encountered */

        return -ENODEV;

    }

    return 0;

}
