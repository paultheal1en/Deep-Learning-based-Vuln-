static void term_backspace(void)

{

    if (term_cmd_buf_index > 0) {

        term_backward_char();

        term_delete_char();

    }

}
