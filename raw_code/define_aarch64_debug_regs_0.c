static void define_aarch64_debug_regs(ARMCPU *cpu)

{

    /* Define breakpoint and watchpoint registers. These do nothing

     * but read as written, for now.

     */

    int i;



    for (i = 0; i < 16; i++) {

        ARMCPRegInfo dbgregs[] = {

            { .name = "DBGBVR", .state = ARM_CP_STATE_AA64,

              .opc0 = 2, .opc1 = 0, .crn = 0, .crm = i, .opc2 = 4,

              .access = PL1_RW,

              .fieldoffset = offsetof(CPUARMState, cp15.dbgbvr[i]) },

            { .name = "DBGBCR", .state = ARM_CP_STATE_AA64,

              .opc0 = 2, .opc1 = 0, .crn = 0, .crm = i, .opc2 = 5,

              .access = PL1_RW,

              .fieldoffset = offsetof(CPUARMState, cp15.dbgbcr[i]) },

            { .name = "DBGWVR", .state = ARM_CP_STATE_AA64,

              .opc0 = 2, .opc1 = 0, .crn = 0, .crm = i, .opc2 = 6,

              .access = PL1_RW,

              .fieldoffset = offsetof(CPUARMState, cp15.dbgwvr[i]) },

            { .name = "DBGWCR", .state = ARM_CP_STATE_AA64,

              .opc0 = 2, .opc1 = 0, .crn = 0, .crm = i, .opc2 = 7,

              .access = PL1_RW,

              .fieldoffset = offsetof(CPUARMState, cp15.dbgwcr[i]) },

               REGINFO_SENTINEL

        };

        define_arm_cp_regs(cpu, dbgregs);

    }

}
