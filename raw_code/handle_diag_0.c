static int handle_diag(S390CPU *cpu, struct kvm_run *run, uint32_t ipb)

{

    int r = 0;

    uint16_t func_code;



    /*

     * For any diagnose call we support, bits 48-63 of the resulting

     * address specify the function code; the remainder is ignored.

     */

    func_code = decode_basedisp_rs(&cpu->env, ipb) & DIAG_KVM_CODE_MASK;

    switch (func_code) {

    case DIAG_IPL:

        kvm_handle_diag_308(cpu, run);

        break;

    case DIAG_KVM_HYPERCALL:

        r = handle_hypercall(cpu, run);

        break;

    case DIAG_KVM_BREAKPOINT:

        r = handle_sw_breakpoint(cpu, run);

        break;

    default:

        DPRINTF("KVM: unknown DIAG: 0x%x\n", func_code);

        r = -1;

        break;

    }



    return r;

}
