static void term_update(void)

{

    int i, delta;



    if (term_cmd_buf_size != term_last_cmd_buf_size ||

        memcmp(term_cmd_buf, term_last_cmd_buf, term_cmd_buf_size) != 0) {

        for(i = 0; i < term_last_cmd_buf_index; i++) {

            term_printf("\033[D");

        }

        term_cmd_buf[term_cmd_buf_size] = '\0';

        term_printf("%s", term_cmd_buf);

        term_printf("\033[K");

        memcpy(term_last_cmd_buf, term_cmd_buf, term_cmd_buf_size);

        term_last_cmd_buf_size = term_cmd_buf_size;

        term_last_cmd_buf_index = term_cmd_buf_size;

    }

    if (term_cmd_buf_index != term_last_cmd_buf_index) {

        delta = term_cmd_buf_index - term_last_cmd_buf_index;

        if (delta > 0) {

            for(i = 0;i < delta; i++) {

                term_printf("\033[C");

            }

        } else {

            delta = -delta;

            for(i = 0;i < delta; i++) {

                term_printf("\033[D");

            }

        }

        term_last_cmd_buf_index = term_cmd_buf_index;

    }

    term_flush();

}
