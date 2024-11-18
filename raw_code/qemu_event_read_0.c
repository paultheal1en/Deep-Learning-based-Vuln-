static void qemu_event_read(void *opaque)

{

    int fd = (intptr_t)opaque;

    ssize_t len;

    char buffer[512];



    /* Drain the notify pipe.  For eventfd, only 8 bytes will be read.  */

    do {

        len = read(fd, buffer, sizeof(buffer));

    } while ((len == -1 && errno == EINTR) || len == sizeof(buffer));

}
