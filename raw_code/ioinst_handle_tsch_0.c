int ioinst_handle_tsch(S390CPU *cpu, uint64_t reg1, uint32_t ipb)

{

    CPUS390XState *env = &cpu->env;

    int cssid, ssid, schid, m;

    SubchDev *sch;

    IRB irb;

    uint64_t addr;

    int cc, irb_len;

    uint8_t ar;



    if (ioinst_disassemble_sch_ident(reg1, &m, &cssid, &ssid, &schid)) {

        program_interrupt(env, PGM_OPERAND, 2);

        return -EIO;

    }

    trace_ioinst_sch_id("tsch", cssid, ssid, schid);

    addr = decode_basedisp_s(env, ipb, &ar);

    if (addr & 3) {

        program_interrupt(env, PGM_SPECIFICATION, 2);

        return -EIO;

    }



    sch = css_find_subch(m, cssid, ssid, schid);

    if (sch && css_subch_visible(sch)) {

        cc = css_do_tsch_get_irb(sch, &irb, &irb_len);

    } else {

        cc = 3;

    }

    /* 0 - status pending, 1 - not status pending, 3 - not operational */

    if (cc != 3) {

        if (s390_cpu_virt_mem_write(cpu, addr, ar, &irb, irb_len) != 0) {

            return -EFAULT;

        }

        css_do_tsch_update_subch(sch);

    } else {

        irb_len = sizeof(irb) - sizeof(irb.emw);

        /* Access exceptions have a higher priority than cc3 */

        if (s390_cpu_virt_mem_check_write(cpu, addr, ar, irb_len) != 0) {

            return -EFAULT;

        }

    }



    setcc(cpu, cc);

    return 0;

}
