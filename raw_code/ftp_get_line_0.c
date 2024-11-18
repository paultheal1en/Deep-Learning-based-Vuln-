static int ftp_get_line(FTPContext *s, char *line, int line_size)

{

    int ch;

    char *q = line;

    int ori_block_flag = s->conn_control_block_flag;



    for (;;) {

        ch = ftp_getc(s);

        if (ch < 0) {

            s->conn_control_block_flag = ori_block_flag;

            return ch;

        }

        if (ch == '\n') {

            /* process line */

            if (q > line && q[-1] == '\r')

                q--;

            *q = '\0';



            s->conn_control_block_flag = ori_block_flag;

            return 0;

        } else {

            s->conn_control_block_flag = 0; /* line need to be finished */

            if ((q - line) < line_size - 1)

                *q++ = ch;

        }

    }

}
