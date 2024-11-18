static void term_show_prompt2(void)

{

    term_printf("(qemu) ");

    fflush(stdout);

    term_last_cmd_buf_index = 0;

    term_last_cmd_buf_size = 0;

    term_esc_state = IS_NORM;

}
