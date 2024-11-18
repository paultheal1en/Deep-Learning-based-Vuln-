void cpu_exec_init(CPUState *cpu, Error **errp)

{

    CPUClass *cc ATTRIBUTE_UNUSED = CPU_GET_CLASS(cpu);



    cpu_list_add(cpu);



#ifndef CONFIG_USER_ONLY

    if (qdev_get_vmsd(DEVICE(cpu)) == NULL) {

        vmstate_register(NULL, cpu->cpu_index, &vmstate_cpu_common, cpu);

    }

    if (cc->vmsd != NULL) {

        vmstate_register(NULL, cpu->cpu_index, cc->vmsd, cpu);

    }

#endif

}
