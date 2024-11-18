int tap_open(char *ifname, int ifname_size, int *vnet_hdr,

             int vnet_hdr_required, int mq_required, Error **errp)

{

    /* FIXME error_setg(errp, ...) on failure */

    char  dev[10]="";

    int fd;

    if( (fd = tap_alloc(dev, sizeof(dev))) < 0 ){

       fprintf(stderr, "Cannot allocate TAP device\n");

       return -1;

    }

    pstrcpy(ifname, ifname_size, dev);

    if (*vnet_hdr) {

        /* Solaris doesn't have IFF_VNET_HDR */

        *vnet_hdr = 0;



        if (vnet_hdr_required && !*vnet_hdr) {

            error_report("vnet_hdr=1 requested, but no kernel "

                         "support for IFF_VNET_HDR available");

            close(fd);

            return -1;

        }

    }

    fcntl(fd, F_SETFL, O_NONBLOCK);

    return fd;

}
