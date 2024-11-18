static uint32_t mb_add_cmdline(MultibootState *s, const char *cmdline)

{

    target_phys_addr_t p = s->offset_cmdlines;

    char *b = (char *)s->mb_buf + p;



    get_opt_value(b, strlen(cmdline) + 1, cmdline);

    s->offset_cmdlines += strlen(b) + 1;

    return s->mb_buf_phys + p;

}
