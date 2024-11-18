static void term_down_char(void)

{

    if (term_hist_entry == TERM_MAX_CMDS - 1 || term_hist_entry == -1)

	return;

    if (term_history[++term_hist_entry] != NULL) {

	pstrcpy(term_cmd_buf, sizeof(term_cmd_buf),

                term_history[term_hist_entry]);

    } else {

	term_hist_entry = -1;

    }

    term_cmd_buf_index = term_cmd_buf_size = strlen(term_cmd_buf);

}
