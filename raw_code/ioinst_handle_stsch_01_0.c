int ioinst_handle_stsch(CPUS390XState *env, uint64_t reg1, uint32_t ipb)

{

    int cssid, ssid, schid, m;

    SubchDev *sch;

    uint64_t addr;

    int cc;

    SCHIB *schib;

    hwaddr len = sizeof(*schib);



    if (ioinst_disassemble_sch_ident(reg1, &m, &cssid, &ssid, &schid)) {

        program_interrupt(env, PGM_OPERAND, 2);

        return -EIO;

    }

    trace_ioinst_sch_id("stsch", cssid, ssid, schid);

    addr = decode_basedisp_s(env, ipb);

    if (addr & 3) {

        program_interrupt(env, PGM_SPECIFICATION, 2);

        return -EIO;

    }

    schib = s390_cpu_physical_memory_map(env, addr, &len, 1);

    if (!schib || len != sizeof(*schib)) {

        program_interrupt(env, PGM_ADDRESSING, 2);

        cc = -EIO;

        goto out;

    }

    sch = css_find_subch(m, cssid, ssid, schid);

    if (sch) {

        if (css_subch_visible(sch)) {

            css_do_stsch(sch, schib);

            cc = 0;

        } else {

            /* Indicate no more subchannels in this css/ss */

            cc = 3;

        }

    } else {

        if (css_schid_final(m, cssid, ssid, schid)) {

            cc = 3; /* No more subchannels in this css/ss */

        } else {

            /* Store an empty schib. */

            memset(schib, 0, sizeof(*schib));

            cc = 0;

        }

    }

out:

    s390_cpu_physical_memory_unmap(env, schib, len, 1);

    return cc;

}
