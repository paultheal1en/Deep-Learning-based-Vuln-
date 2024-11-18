static void socket_sendf(int fd, const char *fmt, va_list ap)

{

    gchar *str;

    size_t size, offset;



    str = g_strdup_vprintf(fmt, ap);

    size = strlen(str);



    offset = 0;

    while (offset < size) {

        ssize_t len;



        len = write(fd, str + offset, size - offset);

        if (len == -1 && errno == EINTR) {

            continue;

        }



        g_assert_no_errno(len);

        g_assert_cmpint(len, >, 0);



        offset += len;

    }

}
