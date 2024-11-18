static void input_linux_event_mouse(void *opaque)

{

    InputLinux *il = opaque;

    struct input_event event;

    int rc;



    for (;;) {

        rc = read(il->fd, &event, sizeof(event));

        if (rc != sizeof(event)) {

            if (rc < 0 && errno != EAGAIN) {

                fprintf(stderr, "%s: read: %s\n", __func__, strerror(errno));

                qemu_set_fd_handler(il->fd, NULL, NULL, NULL);

                close(il->fd);

            }

            break;

        }



        input_linux_handle_mouse(il, &event);

    }

}
