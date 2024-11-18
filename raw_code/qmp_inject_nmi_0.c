void qmp_inject_nmi(Error **errp)

{

#if defined(TARGET_I386)

    CPUState *cs;



    CPU_FOREACH(cs) {

        X86CPU *cpu = X86_CPU(cs);



        if (!cpu->apic_state) {

            cpu_interrupt(cs, CPU_INTERRUPT_NMI);

        } else {

            apic_deliver_nmi(cpu->apic_state);

        }

    }

#else

    nmi_monitor_handle(monitor_get_cpu_index(), errp);

#endif

}
