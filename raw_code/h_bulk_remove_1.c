static target_ulong h_bulk_remove(PowerPCCPU *cpu, sPAPRMachineState *spapr,

                                  target_ulong opcode, target_ulong *args)

{

    CPUPPCState *env = &cpu->env;

    int i;

    target_ulong rc = H_SUCCESS;



    for (i = 0; i < H_BULK_REMOVE_MAX_BATCH; i++) {

        target_ulong *tsh = &args[i*2];

        target_ulong tsl = args[i*2 + 1];

        target_ulong v, r, ret;



        if ((*tsh & H_BULK_REMOVE_TYPE) == H_BULK_REMOVE_END) {

            break;

        } else if ((*tsh & H_BULK_REMOVE_TYPE) != H_BULK_REMOVE_REQUEST) {

            return H_PARAMETER;

        }



        *tsh &= H_BULK_REMOVE_PTEX | H_BULK_REMOVE_FLAGS;

        *tsh |= H_BULK_REMOVE_RESPONSE;



        if ((*tsh & H_BULK_REMOVE_ANDCOND) && (*tsh & H_BULK_REMOVE_AVPN)) {

            *tsh |= H_BULK_REMOVE_PARM;

            return H_PARAMETER;

        }



        ret = remove_hpte(cpu, *tsh & H_BULK_REMOVE_PTEX, tsl,

                          (*tsh & H_BULK_REMOVE_FLAGS) >> 26,

                          &v, &r);



        *tsh |= ret << 60;



        switch (ret) {

        case REMOVE_SUCCESS:

            *tsh |= (r & (HPTE64_R_C | HPTE64_R_R)) << 43;

            break;



        case REMOVE_PARM:

            rc = H_PARAMETER;

            goto exit;



        case REMOVE_HW:

            rc = H_HARDWARE;

            goto exit;

        }

    }

 exit:

    check_tlb_flush(env);



    return rc;

}
