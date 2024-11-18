static void qemu_chr_fe_write_log(CharDriverState *s,

                                  const uint8_t *buf, size_t len)

{

    size_t done = 0;

    ssize_t ret;



    if (s->logfd < 0) {

        return;

    }



    while (done < len) {

        do {

            ret = write(s->logfd, buf + done, len - done);

            if (ret == -1 && errno == EAGAIN) {

                g_usleep(100);

            }

        } while (ret == -1 && errno == EAGAIN);



        if (ret <= 0) {

            return;

        }

        done += ret;

    }

}
