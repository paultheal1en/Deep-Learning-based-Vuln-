static void do_inject_mce(Monitor *mon, const QDict *qdict)

{

    CPUState *cenv;

    int cpu_index = qdict_get_int(qdict, "cpu_index");

    int bank = qdict_get_int(qdict, "bank");

    uint64_t status = qdict_get_int(qdict, "status");

    uint64_t mcg_status = qdict_get_int(qdict, "mcg_status");

    uint64_t addr = qdict_get_int(qdict, "addr");

    uint64_t misc = qdict_get_int(qdict, "misc");

    int broadcast = qdict_get_try_bool(qdict, "broadcast", 0);



    for (cenv = first_cpu; cenv != NULL; cenv = cenv->next_cpu) {

        if (cenv->cpu_index == cpu_index && cenv->mcg_cap) {

            cpu_x86_inject_mce(cenv, bank, status, mcg_status, addr, misc,

                               broadcast);

            break;

        }

    }

}
