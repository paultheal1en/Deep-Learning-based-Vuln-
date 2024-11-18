static void term_delete_char(void)

{

    if (term_cmd_buf_index < term_cmd_buf_size) {

        memmove(term_cmd_buf + term_cmd_buf_index,

                term_cmd_buf + term_cmd_buf_index + 1,

                term_cmd_buf_size - term_cmd_buf_index - 1);

        term_cmd_buf_size--;

    }

}
