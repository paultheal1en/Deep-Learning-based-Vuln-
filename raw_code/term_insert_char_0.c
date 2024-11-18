static void term_insert_char(int ch)

{

    if (term_cmd_buf_index < TERM_CMD_BUF_SIZE) {

        memmove(term_cmd_buf + term_cmd_buf_index + 1,

                term_cmd_buf + term_cmd_buf_index,

                term_cmd_buf_size - term_cmd_buf_index);

        term_cmd_buf[term_cmd_buf_index] = ch;

        term_cmd_buf_size++;

        term_cmd_buf_index++;

    }

}
