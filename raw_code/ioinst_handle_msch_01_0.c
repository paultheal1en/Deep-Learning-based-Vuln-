int ioinst_handle_msch(CPUS390XState *env, uint64_t reg1, uint32_t ipb)

{

    int cssid, ssid, schid, m;

    SubchDev *sch;

    SCHIB *schib;

    uint64_t addr;

    int ret = -ENODEV;

    int cc;

    hwaddr len = sizeof(*schib);



    if (ioinst_disassemble_sch_ident(reg1, &m, &cssid, &ssid, &schid)) {

        program_interrupt(env, PGM_OPERAND, 2);

        return -EIO;

    }

    trace_ioinst_sch_id("msch", cssid, ssid, schid);

    addr = decode_basedisp_s(env, ipb);

    if (addr & 3) {

        program_interrupt(env, PGM_SPECIFICATION, 2);

        return -EIO;

    }

    schib = s390_cpu_physical_memory_map(env, addr, &len, 0);

    if (!schib || len != sizeof(*schib)) {

        program_interrupt(env, PGM_ADDRESSING, 2);

        cc = -EIO;

        goto out;

    }

    if (!ioinst_schib_valid(schib)) {

        program_interrupt(env, PGM_OPERAND, 2);

        cc = -EIO;

        goto out;

    }

    sch = css_find_subch(m, cssid, ssid, schid);

    if (sch && css_subch_visible(sch)) {

        ret = css_do_msch(sch, schib);

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

out:

    s390_cpu_physical_memory_unmap(env, schib, len, 0);

    return cc;

}
