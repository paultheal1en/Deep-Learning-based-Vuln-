void ioinst_handle_stcrw(S390CPU *cpu, uint32_t ipb, uintptr_t ra)

{

    CRW crw;

    uint64_t addr;

    int cc;

    CPUS390XState *env = &cpu->env;

    uint8_t ar;



    addr = decode_basedisp_s(env, ipb, &ar);

    if (addr & 3) {

        s390_program_interrupt(env, PGM_SPECIFICATION, 4, ra);

        return;

    }



    cc = css_do_stcrw(&crw);

    /* 0 - crw stored, 1 - zeroes stored */



    if (s390_cpu_virt_mem_write(cpu, addr, ar, &crw, sizeof(crw)) == 0) {

        setcc(cpu, cc);

    } else if (cc == 0) {

        /* Write failed: requeue CRW since STCRW is a suppressing instruction */

        css_undo_stcrw(&crw);

    }

}
