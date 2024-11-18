int gdbserver_start(const char *device)

{

    GDBState *s;

    char gdbstub_device_name[128];

    CharDriverState *chr = NULL;

    CharDriverState *mon_chr;



    if (!device)

        return -1;

    if (strcmp(device, "none") != 0) {

        if (strstart(device, "tcp:", NULL)) {

            /* enforce required TCP attributes */

            snprintf(gdbstub_device_name, sizeof(gdbstub_device_name),

                     "%s,nowait,nodelay,server", device);

            device = gdbstub_device_name;

        }

#ifndef _WIN32

        else if (strcmp(device, "stdio") == 0) {

            struct sigaction act;



            memset(&act, 0, sizeof(act));

            act.sa_handler = gdb_sigterm_handler;

            sigaction(SIGINT, &act, NULL);

        }

#endif

        chr = qemu_chr_new("gdb", device, NULL);

        if (!chr)

            return -1;




        qemu_chr_add_handlers(chr, gdb_chr_can_receive, gdb_chr_receive,

                              gdb_chr_event, NULL);

    }



    s = gdbserver_state;

    if (!s) {

        s = g_malloc0(sizeof(GDBState));

        gdbserver_state = s;



        qemu_add_vm_change_state_handler(gdb_vm_state_change, NULL);



        /* Initialize a monitor terminal for gdb */

        mon_chr = g_malloc0(sizeof(*mon_chr));

        mon_chr->chr_write = gdb_monitor_write;

        monitor_init(mon_chr, 0);

    } else {

        if (s->chr)

            qemu_chr_delete(s->chr);

        mon_chr = s->mon_chr;

        memset(s, 0, sizeof(GDBState));

    }

    s->c_cpu = first_cpu;

    s->g_cpu = first_cpu;

    s->chr = chr;

    s->state = chr ? RS_IDLE : RS_INACTIVE;

    s->mon_chr = mon_chr;

    s->current_syscall_cb = NULL;



    return 0;

}