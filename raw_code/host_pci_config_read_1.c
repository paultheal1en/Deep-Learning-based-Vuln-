static int host_pci_config_read(int pos, int len, uint32_t val)

{

    char path[PATH_MAX];

    int config_fd;

    ssize_t size = sizeof(path);

    /* Access real host bridge. */

    int rc = snprintf(path, size, "/sys/bus/pci/devices/%04x:%02x:%02x.%d/%s",

                      0, 0, 0, 0, "config");



    if (rc >= size || rc < 0) {

        return -ENODEV;

    }



    config_fd = open(path, O_RDWR);

    if (config_fd < 0) {

        return -ENODEV;

    }



    if (lseek(config_fd, pos, SEEK_SET) != pos) {

        return -errno;

    }

    do {

        rc = read(config_fd, (uint8_t *)&val, len);

    } while (rc < 0 && (errno == EINTR || errno == EAGAIN));

    if (rc != len) {

        return -errno;

    }



    return 0;

}
