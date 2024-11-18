static void vde_from_qemu(void *opaque, const uint8_t *buf, int size)

{

    VDEState *s = opaque;

    int ret;

    for(;;) {

        ret = vde_send(s->vde, (const char *)buf, size, 0);

        if (ret < 0 && errno == EINTR) {

        } else {

            break;

        }

    }

}
