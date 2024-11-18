static target_ulong h_set_mode(PowerPCCPU *cpu, sPAPREnvironment *spapr,

                               target_ulong opcode, target_ulong *args)

{

    CPUState *cs;

    target_ulong mflags = args[0];

    target_ulong resource = args[1];

    target_ulong value1 = args[2];

    target_ulong value2 = args[3];

    target_ulong ret = H_P2;



    if (resource == H_SET_MODE_ENDIAN) {

        if (value1) {

            ret = H_P3;

            goto out;

        }

        if (value2) {

            ret = H_P4;

            goto out;

        }



        switch (mflags) {

        case H_SET_MODE_ENDIAN_BIG:

            CPU_FOREACH(cs) {

                PowerPCCPU *cp = POWERPC_CPU(cs);

                CPUPPCState *env = &cp->env;

                env->spr[SPR_LPCR] &= ~LPCR_ILE;

            }

            ret = H_SUCCESS;

            break;



        case H_SET_MODE_ENDIAN_LITTLE:

            CPU_FOREACH(cs) {

                PowerPCCPU *cp = POWERPC_CPU(cs);

                CPUPPCState *env = &cp->env;

                env->spr[SPR_LPCR] |= LPCR_ILE;

            }

            ret = H_SUCCESS;

            break;



        default:

            ret = H_UNSUPPORTED_FLAG;

        }

    }



out:

    return ret;

}
