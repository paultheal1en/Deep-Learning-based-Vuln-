void monitor_init(CharDriverState *chr, int flags)

{

    static int is_first_init = 1;

    Monitor *mon;



    if (is_first_init) {

        key_timer = qemu_new_timer(vm_clock, release_keys, NULL);

        is_first_init = 0;

    }



    mon = qemu_mallocz(sizeof(*mon));



    mon->chr = chr;

    mon->flags = flags;

    if (flags & MONITOR_USE_READLINE) {

        mon->rs = readline_init(mon, monitor_find_completion);

        monitor_read_command(mon, 0);

    }



    qemu_chr_add_handlers(chr, monitor_can_read, monitor_read, monitor_event,

                          mon);



    LIST_INSERT_HEAD(&mon_list, mon, entry);

    if (!cur_mon || (flags & MONITOR_IS_DEFAULT))

        cur_mon = mon;

}
