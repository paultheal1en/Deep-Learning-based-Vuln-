bool hvf_inject_interrupts(CPUState *cpu_state)

{

    int allow_nmi = !(rvmcs(cpu_state->hvf_fd, VMCS_GUEST_INTERRUPTIBILITY) &

                VMCS_INTERRUPTIBILITY_NMI_BLOCKING);

    X86CPU *x86cpu = X86_CPU(cpu_state);

    CPUX86State *env = &x86cpu->env;



    uint64_t idt_info = rvmcs(cpu_state->hvf_fd, VMCS_IDT_VECTORING_INFO);

    uint64_t info = 0;

    

    if (idt_info & VMCS_IDT_VEC_VALID) {

        uint8_t vector = idt_info & 0xff;

        uint64_t intr_type = idt_info & VMCS_INTR_T_MASK;

        info = idt_info;

        

        uint64_t reason = rvmcs(cpu_state->hvf_fd, VMCS_EXIT_REASON);

        if (intr_type == VMCS_INTR_T_NMI && reason != EXIT_REASON_TASK_SWITCH) {

            allow_nmi = 1;

            vmx_clear_nmi_blocking(cpu_state);

        }

        

        if ((allow_nmi || intr_type != VMCS_INTR_T_NMI)) {

            info &= ~(1 << 12); /* clear undefined bit */

            if (intr_type == VMCS_INTR_T_SWINTR ||

                intr_type == VMCS_INTR_T_PRIV_SWEXCEPTION ||

                intr_type == VMCS_INTR_T_SWEXCEPTION) {

                uint64_t ins_len = rvmcs(cpu_state->hvf_fd,

                                         VMCS_EXIT_INSTRUCTION_LENGTH);

                wvmcs(cpu_state->hvf_fd, VMCS_ENTRY_INST_LENGTH, ins_len);

            }

            if (vector == EXCEPTION_BP || vector == EXCEPTION_OF) {

                /*

                 * VT-x requires #BP and #OF to be injected as software

                 * exceptions.

                 */

                info &= ~VMCS_INTR_T_MASK;

                info |= VMCS_INTR_T_SWEXCEPTION;

                uint64_t ins_len = rvmcs(cpu_state->hvf_fd,

                                         VMCS_EXIT_INSTRUCTION_LENGTH);

                wvmcs(cpu_state->hvf_fd, VMCS_ENTRY_INST_LENGTH, ins_len);

            }

            

            uint64_t err = 0;

            if (idt_info & VMCS_INTR_DEL_ERRCODE) {

                err = rvmcs(cpu_state->hvf_fd, VMCS_IDT_VECTORING_ERROR);

                wvmcs(cpu_state->hvf_fd, VMCS_ENTRY_EXCEPTION_ERROR, err);

            }

            /*printf("reinject  %lx err %d\n", info, err);*/

            wvmcs(cpu_state->hvf_fd, VMCS_ENTRY_INTR_INFO, info);

        };

    }



    if (cpu_state->interrupt_request & CPU_INTERRUPT_NMI) {

        if (allow_nmi && !(info & VMCS_INTR_VALID)) {

            cpu_state->interrupt_request &= ~CPU_INTERRUPT_NMI;

            info = VMCS_INTR_VALID | VMCS_INTR_T_NMI | NMI_VEC;

            wvmcs(cpu_state->hvf_fd, VMCS_ENTRY_INTR_INFO, info);

        } else {

            vmx_set_nmi_window_exiting(cpu_state);

        }

    }



    if (env->hvf_emul->interruptable &&

        (cpu_state->interrupt_request & CPU_INTERRUPT_HARD) &&

        (EFLAGS(env) & IF_MASK) && !(info & VMCS_INTR_VALID)) {

        int line = cpu_get_pic_interrupt(&x86cpu->env);

        cpu_state->interrupt_request &= ~CPU_INTERRUPT_HARD;

        if (line >= 0) {

            wvmcs(cpu_state->hvf_fd, VMCS_ENTRY_INTR_INFO, line |

                  VMCS_INTR_VALID | VMCS_INTR_T_HWINTR);

        }

    }

    if (cpu_state->interrupt_request & CPU_INTERRUPT_HARD) {

        vmx_set_int_window_exiting(cpu_state);

    }

}
