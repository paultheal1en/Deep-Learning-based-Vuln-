static int monitor_read_password(Monitor *mon, ReadLineFunc *readline_func,

                                 void *opaque)

{

    if (mon->rs) {

        readline_start(mon->rs, "Password: ", 1, readline_func, opaque);

        /* prompt is printed on return from the command handler */

        return 0;

    } else {

        monitor_printf(mon, "terminal does not support password prompting\n");

        return -ENOTTY;

    }

}
