void ioinst_handle_msch(S390CPU *cpu, uint64_t reg1, uint32_t ipb)

{

    int cssid, ssid, schid, m;

    SubchDev *sch;

    SCHIB schib;

    uint64_t addr;

    int ret = -ENODEV;

    int cc;

    CPUS390XState *env = &cpu->env;

    uint8_t ar;



    addr = decode_basedisp_s(env, ipb, &ar);

    if (addr & 3) {

        program_interrupt(env, PGM_SPECIFICATION, 2);

        return;

    }

    if (s390_cpu_virt_mem_read(cpu, addr, ar, &schib, sizeof(schib))) {

        return;

    }

    if (ioinst_disassemble_sch_ident(reg1, &m, &cssid, &ssid, &schid) ||

        !ioinst_schib_valid(&schib)) {

        program_interrupt(env, PGM_OPERAND, 2);

        return;

    }

    trace_ioinst_sch_id("msch", cssid, ssid, schid);

    sch = css_find_subch(m, cssid, ssid, schid);

    if (sch && css_subch_visible(sch)) {

        ret = css_do_msch(sch, &schib);

    }

    switch (ret) {

    case -ENODEV:

        cc = 3;

        break;

    case -EBUSY:

        cc = 2;

        break;

    case 0:

        cc = 0;

        break;

    default:

        cc = 1;

        break;

    }

    setcc(cpu, cc);

}
