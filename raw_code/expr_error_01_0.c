static void expr_error(Monitor *mon, const char *msg)

{

    monitor_printf(mon, "%s\n", msg);

    siglongjmp(expr_env, 1);

}
