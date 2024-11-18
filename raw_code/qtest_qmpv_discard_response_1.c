void qtest_qmpv_discard_response(QTestState *s, const char *fmt, va_list ap)

{

    bool has_reply = false;

    int nesting = 0;



    /* Send QMP request */

    socket_sendf(s->qmp_fd, fmt, ap);



    /* Receive reply */

    while (!has_reply || nesting > 0) {

        ssize_t len;

        char c;



        len = read(s->qmp_fd, &c, 1);

        if (len == -1 && errno == EINTR) {

            continue;

        }



        if (len == -1 || len == 0) {

            fprintf(stderr, "Broken pipe\n");

            exit(1);

        }



        switch (c) {

        case '{':

            nesting++;

            has_reply = true;

            break;

        case '}':

            nesting--;

            break;

        }

    }

}
