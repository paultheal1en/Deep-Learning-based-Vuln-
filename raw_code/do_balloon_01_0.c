static void do_balloon(Monitor *mon, const QDict *qdict, QObject **ret_data)

{

    int value = qdict_get_int(qdict, "value");

    ram_addr_t target = value;

    qemu_balloon(target << 20);

}
