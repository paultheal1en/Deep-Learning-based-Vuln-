void openrisc_cpu_do_interrupt(CPUState *cs)

{

#ifndef CONFIG_USER_ONLY

    OpenRISCCPU *cpu = OPENRISC_CPU(cs);

    CPUOpenRISCState *env = &cpu->env;



    env->epcr = env->pc;

    if (env->flags & D_FLAG) {

        env->flags &= ~D_FLAG;

        env->sr |= SR_DSX;

        env->epcr -= 4;




    if (cs->exception_index == EXCP_SYSCALL) {

        env->epcr += 4;









    /* For machine-state changed between user-mode and supervisor mode,

       we need flush TLB when we enter&exit EXCP.  */

    tlb_flush(cs);



    env->esr = env->sr;

    env->sr &= ~SR_DME;

    env->sr &= ~SR_IME;

    env->sr |= SR_SM;

    env->sr &= ~SR_IEE;

    env->sr &= ~SR_TEE;

    env->tlb->cpu_openrisc_map_address_data = &cpu_openrisc_get_phys_nommu;

    env->tlb->cpu_openrisc_map_address_code = &cpu_openrisc_get_phys_nommu;



    if (cs->exception_index > 0 && cs->exception_index < EXCP_NR) {

        env->pc = (cs->exception_index << 8);


        cpu_abort(cs, "Unhandled exception 0x%x\n", cs->exception_index);


#endif



    cs->exception_index = -1;
