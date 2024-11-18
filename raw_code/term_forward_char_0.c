static void term_forward_char(void)

{

    if (term_cmd_buf_index < term_cmd_buf_size) {

        term_cmd_buf_index++;

    }

}
