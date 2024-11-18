static int cdrom_probe_device(const char *filename)

{

    int fd, ret;

    int prio = 0;



    if (strstart(filename, "/dev/cd", NULL))

        prio = 50;



    fd = open(filename, O_RDONLY | O_NONBLOCK);

    if (fd < 0) {

        goto out;

    }



    /* Attempt to detect via a CDROM specific ioctl */

    ret = ioctl(fd, CDROM_DRIVE_STATUS, CDSL_CURRENT);

    if (ret >= 0)

        prio = 100;



    close(fd);

out:

    return prio;

}
