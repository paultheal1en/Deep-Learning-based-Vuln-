void ioinst_handle_chsc(S390CPU *cpu, uint32_t ipb)

{

    ChscReq *req;

    ChscResp *res;

    uint64_t addr;

    int reg;

    uint16_t len;

    uint16_t command;

    CPUS390XState *env = &cpu->env;

    uint8_t buf[TARGET_PAGE_SIZE];



    trace_ioinst("chsc");

    reg = (ipb >> 20) & 0x00f;

    addr = env->regs[reg];

    /* Page boundary? */

    if (addr & 0xfff) {

        program_interrupt(env, PGM_SPECIFICATION, 2);

        return;

    }

    /*

     * Reading sizeof(ChscReq) bytes is currently enough for all of our

     * present CHSC sub-handlers ... if we ever need more, we should take

     * care of req->len here first.

     */

    if (s390_cpu_virt_mem_read(cpu, addr, reg, buf, sizeof(ChscReq))) {

        return;

    }

    req = (ChscReq *)buf;

    len = be16_to_cpu(req->len);

    /* Length field valid? */

    if ((len < 16) || (len > 4088) || (len & 7)) {

        program_interrupt(env, PGM_OPERAND, 2);

        return;

    }

    memset((char *)req + len, 0, TARGET_PAGE_SIZE - len);

    res = (void *)((char *)req + len);

    command = be16_to_cpu(req->command);

    trace_ioinst_chsc_cmd(command, len);

    switch (command) {

    case CHSC_SCSC:

        ioinst_handle_chsc_scsc(req, res);

        break;

    case CHSC_SCPD:

        ioinst_handle_chsc_scpd(req, res);

        break;

    case CHSC_SDA:

        ioinst_handle_chsc_sda(req, res);

        break;

    case CHSC_SEI:

        ioinst_handle_chsc_sei(req, res);

        break;

    default:

        ioinst_handle_chsc_unimplemented(res);

        break;

    }



    if (!s390_cpu_virt_mem_write(cpu, addr + len, reg, res,

                                 be16_to_cpu(res->len))) {

        setcc(cpu, 0);    /* Command execution complete */

    }

}
