static int ftp_getc(FTPContext *s)

{

    int len;

    if (s->control_buf_ptr >= s->control_buf_end) {

        if (s->conn_control_block_flag)

            return AVERROR_EXIT;

        len = ffurl_read(s->conn_control, s->control_buffer, CONTROL_BUFFER_SIZE);

        if (len < 0) {

            return len;

        } else if (!len) {

            return -1;

        } else {

            s->control_buf_ptr = s->control_buffer;

            s->control_buf_end = s->control_buffer + len;

        }

    }

    return *s->control_buf_ptr++;

}
