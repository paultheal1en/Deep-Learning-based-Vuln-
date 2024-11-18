CpuInfoList *qmp_query_cpus(Error **errp)

{

    MachineState *ms = MACHINE(qdev_get_machine());

    MachineClass *mc = MACHINE_GET_CLASS(ms);

    CpuInfoList *head = NULL, *cur_item = NULL;

    CPUState *cpu;



    CPU_FOREACH(cpu) {

        CpuInfoList *info;

#if defined(TARGET_I386)

        X86CPU *x86_cpu = X86_CPU(cpu);

        CPUX86State *env = &x86_cpu->env;

#elif defined(TARGET_PPC)

        PowerPCCPU *ppc_cpu = POWERPC_CPU(cpu);

        CPUPPCState *env = &ppc_cpu->env;

#elif defined(TARGET_SPARC)

        SPARCCPU *sparc_cpu = SPARC_CPU(cpu);

        CPUSPARCState *env = &sparc_cpu->env;

#elif defined(TARGET_MIPS)

        MIPSCPU *mips_cpu = MIPS_CPU(cpu);

        CPUMIPSState *env = &mips_cpu->env;

#elif defined(TARGET_TRICORE)

        TriCoreCPU *tricore_cpu = TRICORE_CPU(cpu);

        CPUTriCoreState *env = &tricore_cpu->env;

#endif



        cpu_synchronize_state(cpu);



        info = g_malloc0(sizeof(*info));

        info->value = g_malloc0(sizeof(*info->value));

        info->value->CPU = cpu->cpu_index;

        info->value->current = (cpu == first_cpu);

        info->value->halted = cpu->halted;

        info->value->qom_path = object_get_canonical_path(OBJECT(cpu));

        info->value->thread_id = cpu->thread_id;

#if defined(TARGET_I386)

        info->value->arch = CPU_INFO_ARCH_X86;

        info->value->u.x86.pc = env->eip + env->segs[R_CS].base;

#elif defined(TARGET_PPC)

        info->value->arch = CPU_INFO_ARCH_PPC;

        info->value->u.ppc.nip = env->nip;

#elif defined(TARGET_SPARC)

        info->value->arch = CPU_INFO_ARCH_SPARC;

        info->value->u.q_sparc.pc = env->pc;

        info->value->u.q_sparc.npc = env->npc;

#elif defined(TARGET_MIPS)

        info->value->arch = CPU_INFO_ARCH_MIPS;

        info->value->u.q_mips.PC = env->active_tc.PC;

#elif defined(TARGET_TRICORE)

        info->value->arch = CPU_INFO_ARCH_TRICORE;

        info->value->u.tricore.PC = env->PC;

#else

        info->value->arch = CPU_INFO_ARCH_OTHER;

#endif










        /* XXX: waiting for the qapi to support GSList */

        if (!cur_item) {

            head = cur_item = info;

        } else {

            cur_item->next = info;

            cur_item = info;





    return head;
