void hmp_info_local_apic(Monitor *mon, const QDict *qdict)

{

    x86_cpu_dump_local_apic_state(mon_get_cpu(), (FILE *)mon, monitor_fprintf,

                                  CPU_DUMP_FPU);

}
