static void s390_cpu_realizefn(DeviceState *dev, Error **errp)

{

    CPUState *cs = CPU(dev);

    S390CPUClass *scc = S390_CPU_GET_CLASS(dev);

    S390CPU *cpu = S390_CPU(dev);

    CPUS390XState *env = &cpu->env;

    Error *err = NULL;



    cpu_exec_init(cs, &err);

    if (err != NULL) {

        error_propagate(errp, err);

        return;

    }



#if !defined(CONFIG_USER_ONLY)

    qemu_register_reset(s390_cpu_machine_reset_cb, cpu);

#endif

    env->cpu_num = scc->next_cpu_id++;

    s390_cpu_gdb_init(cs);

    qemu_init_vcpu(cs);

#if !defined(CONFIG_USER_ONLY)

    run_on_cpu(cs, s390_do_cpu_full_reset, cs);

#else

    cpu_reset(cs);

#endif



    scc->parent_realize(dev, errp);

}
