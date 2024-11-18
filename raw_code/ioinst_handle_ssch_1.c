void ioinst_handle_ssch(S390CPU *cpu, uint64_t reg1, uint32_t ipb)

{

    int cssid, ssid, schid, m;

    SubchDev *sch;

    ORB orig_orb, orb;

    uint64_t addr;

    int ret = -ENODEV;

    int cc;

    CPUS390XState *env = &cpu->env;

    uint8_t ar;



    addr = decode_basedisp_s(env, ipb, &ar);

    if (addr & 3) {

        program_interrupt(env, PGM_SPECIFICATION, 4);

        return;

    }

    if (s390_cpu_virt_mem_read(cpu, addr, ar, &orig_orb, sizeof(orb))) {

        return;

    }

    copy_orb_from_guest(&orb, &orig_orb);

    if (ioinst_disassemble_sch_ident(reg1, &m, &cssid, &ssid, &schid) ||

        !ioinst_orb_valid(&orb)) {

        program_interrupt(env, PGM_OPERAND, 4);

        return;

    }

    trace_ioinst_sch_id("ssch", cssid, ssid, schid);

    sch = css_find_subch(m, cssid, ssid, schid);

    if (sch && css_subch_visible(sch)) {

        ret = css_do_ssch(sch, &orb);

    }

    switch (ret) {

    case -ENODEV:

        cc = 3;

        break;

    case -EBUSY:

        cc = 2;

        break;

    case -EFAULT:

        /*

         * TODO:

         * I'm wondering whether there is something better

         * to do for us here (like setting some device or

         * subchannel status).

         */

        program_interrupt(env, PGM_ADDRESSING, 4);

        return;

    case 0:

        cc = 0;

        break;

    default:

        cc = 1;

        break;

    }

    setcc(cpu, cc);

}
