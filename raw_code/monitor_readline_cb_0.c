static void monitor_readline_cb(void *opaque, const char *input)

{

    pstrcpy(monitor_readline_buf, monitor_readline_buf_size, input);

    monitor_readline_started = 0;

}
